"""Run with:  python -m pytest -q"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.freight_rate import config
from src.freight_rate.data import Cleaner, flag_label_outliers
from src.freight_rate.features import build_features


def _frame(n=200, seed=0):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "load_id": [f"X-{i}" for i in range(n)],
        "pickup": "A", "delivery": "B",
        "pickup_lat": 35.0, "pickup_lon": -90.0, "delivery_lat": 40.0, "delivery_lon": -85.0,
        "distance": rng.uniform(100, 2000, n),
        "equipment": rng.choice(list(config.EQUIPMENT_CODES), n),
        "weight": rng.uniform(10000, 45000, n),
        "date": pd.to_datetime("2025-03-01") + pd.to_timedelta(rng.randint(0, 60, n), "D"),
        "market_index": rng.uniform(0.9, 1.2, n),
        "quote_signal": rng.uniform(1.8, 2.2, n),
    })
    df["posted_rate"] = 2.0 * df["distance"] ** 0.9 * df["market_index"] ** 0.15 * np.exp(rng.normal(0, 0.03, n))
    return df


def test_cleaner_fixes_negative_and_missing_values():
    df = _frame()
    df.loc[0, "weight"] = -30000
    df.loc[1, "weight"] = np.nan
    df.loc[2, "market_index"] = np.nan
    out = Cleaner().fit(df).transform(df)
    assert out.loc[0, "weight"] == 30000
    assert out["weight"].notna().all() and (out["weight"] > 0).all()
    assert out["market_index"].notna().all()
    assert out.loc[1, "weight_was_missing"] == 1 and out.loc[0, "weight_was_negative"] == 1


def test_outlier_flag_catches_corrupted_labels_only():
    df = Cleaner().fit(_frame()).transform(_frame())
    df.loc[[5, 6], "posted_rate"] *= 4
    df.loc[[7], "posted_rate"] *= 0.25
    flags = flag_label_outliers(df)
    assert set(np.where(flags)[0]) == {5, 6, 7}


def test_calendar_features():
    df = build_features(_frame().assign(date=pd.to_datetime(["2025-12-31"] * 200)))
    assert (df["days_to_quarter_end"] == 0).all()
    assert (df["quarter"] == 4).all()
    df = build_features(_frame().assign(date=pd.to_datetime(["2025-11-01"] * 200)))
    assert (df["days_to_quarter_end"] == 60).all()


def test_all_model_features_present():
    df = build_features(Cleaner().fit(_frame()).transform(_frame()))
    assert not df[config.FEATURES].isna().any().any()


def test_unknown_equipment_raises():
    with pytest.raises(ValueError):
        build_features(_frame().assign(equipment="Tanker"))


@pytest.mark.skipif(not config.PREDICTIONS_PATH.exists(), reason="run the pipeline first")
def test_submission_file_format():
    sub = pd.read_csv(config.PREDICTIONS_PATH)
    template = pd.read_csv(config.TEMPLATE_PATH)
    assert list(sub.columns) == ["load_id", "predicted_rate"]
    assert len(sub) == 12_000 and sub["load_id"].is_unique
    assert set(sub["load_id"]) == set(template["load_id"])
    assert np.isfinite(sub["predicted_rate"]).all() and (sub["predicted_rate"] > 0).all()


@pytest.mark.skipif(not config.PREDICTIONS_PATH.exists(), reason="run the pipeline first")
def test_december_file_format():
    dec = pd.read_csv(config.DECEMBER_PATH)
    assert list(dec.columns) == ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]
    assert len(dec) == 31 and (dec["predicted_rate"] > 0).all()
