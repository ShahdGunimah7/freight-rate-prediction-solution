"""EDA and validation figures used in the report.  python -m src.freight_rate.figures"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .data import Cleaner, flag_label_outliers, load_csv

INK, MUTED, GRID = "#1F2A30", "#5B6B70", "#E3E8EA"
COLORS = {"Dry Van": "#2B6CB0", "Reefer": "#C05621", "Flatbed": "#2F855A"}
ACCENT, BAD = "#064A56", "#C53030"


def _style(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#9DAFB3")
    ax.tick_params(colors=MUTED)


def _save(fig, name):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _log_residuals(df: pd.DataFrame) -> np.ndarray:
    keep = ~df["is_label_outlier"].to_numpy()
    X = np.column_stack([np.ones(len(df)), np.log(df["distance"]), np.log(df["market_index"]),
                         (df["equipment"] == "Reefer").astype(float), (df["equipment"] == "Flatbed").astype(float),
                         np.log(df["weight"])])
    y = np.log(df[config.TARGET].to_numpy())
    coef, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
    return y - X @ coef


def main() -> None:
    raw = load_csv(config.TRAIN_PATH)
    df = Cleaner().fit(raw).transform(raw)
    df["is_label_outlier"] = flag_label_outliers(df)
    df["resid"] = _log_residuals(df)
    clean = df[~df["is_label_outlier"]]

    # 1. Rate vs distance
    fig, ax = plt.subplots(figsize=(8, 4.6))
    sample = clean.sample(6000, random_state=1)
    for eq, c in COLORS.items():
        s = sample[sample["equipment"] == eq]
        ax.scatter(s["distance"], s[config.TARGET], s=4, alpha=0.35, color=c, label=eq)
    out = df[df["is_label_outlier"]]
    ax.scatter(out["distance"], out[config.TARGET], s=10, color=BAD, marker="x", label="Corrupted label (removed)")
    ax.set_yscale("log")
    ax.legend(frameon=False, markerscale=3)
    _style(ax, "Posted rate vs distance", "Distance (miles)", "Posted rate ($, log scale)")
    _save(fig, "01_rate_vs_distance.png")

    # 2. Residual histogram showing the outlier gap
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.hist(np.exp(df["resid"]), bins=np.logspace(np.log10(0.1), np.log10(8), 120), color=ACCENT)
    ax.set_xscale("log")
    ax.set_yscale("log")
    for x in (np.exp(-config.OUTLIER_LOG_THRESHOLD), np.exp(config.OUTLIER_LOG_THRESHOLD)):
        ax.axvline(x, color=BAD, linestyle="--", linewidth=1)
    _style(ax, "Actual / expected rate: corrupted labels form separate clusters", "Actual rate / robust-fit rate (log)", "Loads (log)")
    _save(fig, "02_label_outliers.png")

    # 3. Calendar effect: residual by day (quarter-end ramp)
    daily = clean.groupby("date")["resid"].mean()
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(daily.index, 100 * daily.values, color=ACCENT, linewidth=1.4)
    for q in ["2025-03-31", "2025-06-30", "2025-09-30"]:
        ax.axvline(pd.Timestamp(q), color=MUTED, linestyle=":", linewidth=1)
    _style(ax, "Rates ramp up into every quarter end, then reset", "", "Daily mean residual (%)")
    _save(fig, "03_quarter_end_ramp.png")

    # 4. Market index
    mi = df.groupby("date")["market_index"].mean()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [2.3, 1]})
    axes[0].plot(mi.index, mi.values, color=ACCENT, linewidth=1)
    _style(axes[0], "Daily mean market_index (train)", "", "market_index")
    dow = mi.groupby(mi.index.dayofweek).mean()
    axes[1].bar(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], dow.values, color=ACCENT)
    axes[1].set_ylim(dow.min() * 0.95, dow.max() * 1.02)
    _style(axes[1], "Weekly cycle", "", "")
    _save(fig, "04_market_index.png")

    # 5. Backtest results
    with open(config.REPORTS_DIR / "metrics.json") as fh:
        metrics = json.load(fh)
    folds = list(metrics["backtests"])
    models = list(metrics["backtests"][folds[0]]["models"])
    fig, ax = plt.subplots(figsize=(9, 3.9))
    width = 0.38
    y = np.arange(len(models))
    for i, f in enumerate(folds):
        vals = [metrics["backtests"][f]["models"][m]["clean_labels"]["MAPE_%"] for m in models]
        bars = ax.barh(y + (i - 0.5) * width, vals, height=width, color=[ACCENT, "#7FB3BC"][i], label=f)
        ax.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=8, color=MUTED)
    ax.set_yticks(y, [m.replace(" (final)", "\n(final)") for m in models], fontsize=8)
    ax.invert_yaxis()
    ax.legend(frameon=False)
    _style(ax, "Time-based backtest: MAPE on clean labels (lower is better)", "MAPE (%)", "")
    ax.grid(axis="x", color=GRID)
    _save(fig, "05_backtest_mape.png")

    # 6. Predicted vs actual (backtests)
    bt = pd.read_csv(config.REPORTS_DIR / "backtest_predictions.csv")
    bt = bt[~bt["is_label_outlier"]]
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.scatter(bt[config.TARGET], bt["predicted_rate"], s=3, alpha=0.25, color=ACCENT)
    lim = [bt[config.TARGET].min(), bt[config.TARGET].max()]
    ax.plot(lim, lim, color=BAD, linewidth=1)
    _style(ax, "Backtest: predicted vs actual", "Actual rate ($)", "Predicted rate ($)")
    _save(fig, "06_pred_vs_actual.png")
    print(f"Figures written to {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
