"""End-to-end pipeline: clean -> backtest -> train final model -> predict.

Run from the repository root:
    python -m src.freight_rate.pipeline      (or)      python run_pipeline.py
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config
from .data import Cleaner, flag_label_outliers, load_csv
from .features import build_features
from .metrics import regression_metrics
from .model import FreightRateModel


# ----------------------------------------------------------------------------
# Data preparation
# ----------------------------------------------------------------------------
def prepare_training_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, Cleaner]:
    cleaner = Cleaner().fit(raw)
    df = cleaner.transform(raw)
    df["is_label_outlier"] = flag_label_outliers(df)
    return build_features(df), cleaner


def city_coordinates(*frames: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for frame in frames:
        for side in ("pickup", "delivery"):
            part = frame[[side, f"{side}_lat", f"{side}_lon"]]
            parts.append(part.set_axis(["city", "lat", "lon"], axis=1))
    return pd.concat(parts).drop_duplicates("city").set_index("city")


# ----------------------------------------------------------------------------
# Baselines used for comparison in the backtests
# ----------------------------------------------------------------------------
def baseline_rate_per_mile(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    rpm = (train[config.TARGET] / train["distance"]).groupby(train["equipment"]).median()
    return test["equipment"].map(rpm).to_numpy() * test["distance"].to_numpy()


def baseline_log_linear(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    def design(df):
        return np.column_stack(
            [
                np.ones(len(df)),
                df["log_distance"],
                np.log(df["market_index"]),
                np.log(df["weight"]),
                df["quote_signal"],
                (df["equipment"] == "Reefer").astype(float),
                (df["equipment"] == "Flatbed").astype(float),
            ]
        )
    coef, *_ = np.linalg.lstsq(design(train), np.log(train[config.TARGET]), rcond=None)
    return np.exp(design(test) @ coef)


def lightgbm_unconstrained(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    params = dict(config.LGB_PARAMS, seed=config.RANDOM_SEED)
    booster = lgb.train(params, lgb.Dataset(train[config.FEATURES], np.log(train[config.TARGET])), config.NUM_BOOST_ROUND)
    return np.exp(booster.predict(test[config.FEATURES]))


def final_model(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    return FreightRateModel().fit(train, train[config.TARGET]).predict(test)


CANDIDATES = {
    "Baseline: median rate-per-mile x distance": baseline_rate_per_mile,
    "Baseline: log-linear regression": baseline_log_linear,
    "LightGBM (unconstrained time features)": lightgbm_unconstrained,
    "LightGBM + additive calendar effects (final)": final_model,
}


def _evaluate(test: pd.DataFrame, pred: np.ndarray) -> dict:
    clean = ~test["is_label_outlier"].to_numpy()
    return {
        "clean_labels": regression_metrics(test[config.TARGET].to_numpy()[clean], pred[clean]),
        "all_labels": regression_metrics(test[config.TARGET].to_numpy(), pred),
    }


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------
def run_backtests(raw: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Time-based backtests. Cleaning statistics and outlier flags are re-fit on
    each fold's training window only, so nothing from the test window leaks in."""
    results, fold_predictions = {}, []
    for fold in config.BACKTEST_FOLDS:
        train_raw = raw[raw["date"] <= fold["train_end"]]
        test_raw = raw[(raw["date"] >= fold["test_start"]) & (raw["date"] <= fold["test_end"])]
        train, cleaner = prepare_training_frame(train_raw)
        test = cleaner.transform(test_raw)
        # outlier flag on the test window is used ONLY to report clean-label metrics
        test["is_label_outlier"] = flag_label_outliers(test)
        test = build_features(test)
        train_clean = train[~train["is_label_outlier"]]

        fold_res = {}
        for name, fn in CANDIDATES.items():
            pred = fn(train_clean, test)
            fold_res[name] = _evaluate(test, pred)
            if name.endswith("(final)"):
                fold_predictions.append(test.assign(predicted_rate=pred, fold=fold["name"]))
        results[fold["name"]] = {
            "train_window": f"2025-01-01 .. {fold['train_end']}",
            "test_window": f"{fold['test_start']} .. {fold['test_end']}",
            "n_train": int(len(train_clean)),
            "n_test": int(len(test)),
            "models": fold_res,
        }
        print(f"{fold['name']}: final model clean MAPE = {fold_res[list(CANDIDATES)[-1]]['clean_labels']['MAPE_%']:.2f}%")
    return results, pd.concat(fold_predictions)


def run_random_split_comparison(train_all: pd.DataFrame) -> dict:
    """Shows why a random split is the wrong choice: it leaks future market
    conditions into training and gives an optimistic score."""
    clean = train_all[~train_all["is_label_outlier"]]
    rng = np.random.RandomState(config.RANDOM_SEED)
    test_mask = rng.rand(len(clean)) < 0.2
    pred = final_model(clean[~test_mask], clean[test_mask])
    return regression_metrics(clean.loc[test_mask, config.TARGET], pred)


def unseen_city_check(train_all: pd.DataFrame, n_cities: int = 8) -> dict:
    """Validation contains 8 cities never seen in training. Simulate that by
    holding out 8 random cities from the training window of Fold B."""
    fold = config.BACKTEST_FOLDS[-1]
    rng = np.random.RandomState(config.RANDOM_SEED)
    cities = sorted(set(train_all["pickup"]) | set(train_all["delivery"]))
    held = set(rng.choice(cities, n_cities, replace=False))
    touches = train_all["pickup"].isin(held) | train_all["delivery"].isin(held)
    clean = train_all[~train_all["is_label_outlier"]]
    train = clean[(clean["date"] <= fold["train_end"]) & ~touches.loc[clean.index]]
    test = clean[(clean["date"] >= fold["test_start"]) & (clean["date"] <= fold["test_end"])]
    pred = pd.Series(final_model(train, test), index=test.index)
    is_new = touches.loc[test.index]
    return {
        "held_out_cities": sorted(held),
        "rows_with_unseen_city": regression_metrics(test.loc[is_new, config.TARGET], pred[is_new]),
        "rows_with_seen_cities": regression_metrics(test.loc[~is_new, config.TARGET], pred[~is_new]),
    }


# ----------------------------------------------------------------------------
# Final predictions
# ----------------------------------------------------------------------------
def build_december_frame(december: pd.DataFrame, validation_clean: pd.DataFrame, coords: pd.DataFrame) -> pd.DataFrame:
    """The December chart inputs only fix lane, distance, equipment, weight and
    date. The market signals for each day are taken from the validation file,
    which contains real Nov/Dec market_index and quote_signal values: we use the
    daily mean market_index and daily median quote_signal."""
    daily = validation_clean.groupby("date").agg(market_index=("market_index", "mean"), quote_signal=("quote_signal", "median"))
    df = december.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.join(daily, on="date")
    if df[["market_index", "quote_signal"]].isna().any().any():
        raise ValueError("Missing daily market signals for some December dates")
    for side in ("pickup", "delivery"):
        df[f"{side}_lat"] = df[side].map(coords["lat"])
        df[f"{side}_lon"] = df[side].map(coords["lon"])
    if df[["pickup_lat", "delivery_lat"]].isna().any().any():
        raise ValueError("Unknown city in December inputs")
    return build_features(df)


def main() -> None:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_train = load_csv(config.TRAIN_PATH)
    raw_validation = load_csv(config.VALIDATION_PATH)
    template = pd.read_csv(config.TEMPLATE_PATH)
    december_raw = pd.read_csv(config.DECEMBER_PATH)

    print("1/4  Cleaning and feature engineering")
    train_all, cleaner = prepare_training_frame(raw_train)
    data_quality = {
        "rows": int(len(raw_train)),
        "negative_weight": int((raw_train["weight"] < 0).sum()),
        "missing_weight": int(raw_train["weight"].isna().sum()),
        "weight_at_47500_cap": int((raw_train["weight"].abs() == 47500).sum()),
        "missing_market_index": int(raw_train["market_index"].isna().sum()),
        "label_outliers_removed": int(train_all["is_label_outlier"].sum()),
        "validation_negative_weight": int((raw_validation["weight"] < 0).sum()),
        "validation_missing_weight": int(raw_validation["weight"].isna().sum()),
        "validation_weight_at_47500_cap": int((raw_validation["weight"].abs() == 47500).sum()),
        "validation_missing_market_index": int(raw_validation["market_index"].isna().sum()),
        "validation_cities_not_in_training": sorted(
            (set(raw_validation["pickup"]) | set(raw_validation["delivery"]))
            - (set(raw_train["pickup"]) | set(raw_train["delivery"]))
        ),
    }

    print("2/4  Time-based backtests")
    backtests, fold_predictions = run_backtests(raw_train)
    random_split = run_random_split_comparison(train_all)
    unseen = unseen_city_check(train_all)

    print("3/4  Training final model on Jan-Oct (clean labels)")
    train_clean = train_all[~train_all["is_label_outlier"]]
    model = FreightRateModel().fit(train_clean, train_clean[config.TARGET])
    model.save(config.MODELS_DIR / "freight_rate_model.pkl")

    print("4/4  Predicting validation and December chart inputs")
    validation = build_features(cleaner.transform(raw_validation))
    validation["predicted_rate"] = model.predict(validation)
    submission = template[["load_id"]].merge(validation[["load_id", "predicted_rate"]], on="load_id", how="left", validate="one_to_one")
    if submission["predicted_rate"].isna().any() or len(submission) != len(template):
        raise RuntimeError("Prediction/template mismatch")
    submission["predicted_rate"] = submission["predicted_rate"].round(2)
    submission.to_csv(config.PREDICTIONS_PATH, index=False)

    coords = city_coordinates(raw_train, raw_validation)
    december = build_december_frame(december_raw, validation, coords)
    december_out = december_raw.copy()
    december_out["predicted_rate"] = np.round(model.predict(december), 2)
    december_out.to_csv(config.DECEMBER_PATH, index=False)

    summary = {
        "data_quality": data_quality,
        "backtests": backtests,
        "random_split_80_20_for_comparison": random_split,
        "unseen_city_check": unseen,
        "feature_importance_gain_share": model.feature_importance().round(5).to_dict(),
        "validation_prediction_summary": submission["predicted_rate"].describe().round(2).to_dict(),
        "december_predictions": december_out[["date", "predicted_rate"]].assign(date=lambda d: d["date"].astype(str)).to_dict("records"),
    }
    with open(config.REPORTS_DIR / "metrics.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    fold_predictions[["load_id", "date", "fold", "equipment", "distance", config.TARGET, "predicted_rate", "is_label_outlier"]].to_csv(
        config.REPORTS_DIR / "backtest_predictions.csv", index=False
    )
    print(f"Wrote {config.PREDICTIONS_PATH.name} ({len(submission):,} rows) and filled {config.DECEMBER_PATH.name}")


if __name__ == "__main__":
    main()
