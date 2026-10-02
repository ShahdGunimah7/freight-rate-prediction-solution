"""Loading and cleaning of the raw CSV files."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


@dataclass
class Cleaner:
    """Fits cleaning statistics on training data and applies them to any frame.

    Issues handled:
    * negative weights  -> sign error, absolute value taken
    * missing weight    -> median weight of the same equipment type (training stats)
    * missing market_index -> mean market_index of the same date (it is a daily
      market signal; within-day std is ~0.025 vs ~0.17 across days)
    """

    weight_median_by_equipment: dict = field(default_factory=dict)
    global_weight_median: float = float("nan")
    global_market_index_median: float = float("nan")

    def fit(self, train: pd.DataFrame) -> "Cleaner":
        weight = train["weight"].abs()
        self.weight_median_by_equipment = weight.groupby(train["equipment"]).median().to_dict()
        self.global_weight_median = float(weight.median())
        self.global_market_index_median = float(train["market_index"].median())
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        df = frame.copy()
        df["weight_was_negative"] = (df["weight"] < 0).astype(int)
        df["weight_was_missing"] = df["weight"].isna().astype(int)
        df["weight"] = df["weight"].abs()
        fill = df["equipment"].map(self.weight_median_by_equipment).fillna(self.global_weight_median)
        df["weight"] = df["weight"].fillna(fill)

        df["market_index_was_missing"] = df["market_index"].isna().astype(int)
        daily_mean = df.groupby("date")["market_index"].transform("mean")
        df["market_index"] = df["market_index"].fillna(daily_mean).fillna(self.global_market_index_median)
        return df


def flag_label_outliers(df: pd.DataFrame, threshold: float = config.OUTLIER_LOG_THRESHOLD) -> pd.Series:
    """Flag corrupted labels with an iteratively re-weighted log-linear fit.

    log(rate) ~ log(distance) + log(market_index) + equipment + log(weight).
    Points whose |residual| exceeds `threshold` are excluded from the next fit
    so the outliers cannot pull the reference line towards themselves.
    """
    X = np.column_stack(
        [
            np.ones(len(df)),
            np.log(df["distance"]),
            np.log(df["market_index"]),
            (df["equipment"] == "Reefer").astype(float),
            (df["equipment"] == "Flatbed").astype(float),
            np.log(df["weight"]),
        ]
    )
    y = np.log(df[config.TARGET].to_numpy())
    keep = np.ones(len(df), dtype=bool)
    for _ in range(5):
        coef, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
        residual = y - X @ coef
        new_keep = np.abs(residual) <= threshold
        if (new_keep == keep).all():
            break
        keep = new_keep
    return pd.Series(~keep, index=df.index, name="is_label_outlier")
