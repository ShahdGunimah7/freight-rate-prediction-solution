"""Feature engineering. Pure functions of a single row/date -> no leakage."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def initial_bearing(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Compass bearing (degrees) from pickup to delivery."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    x = np.sin(lon2 - lon1) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    return np.degrees(np.arctan2(x, y))


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    date = pd.to_datetime(df["date"])
    quarter_end = date.dt.to_period("Q").dt.end_time.dt.normalize()
    df["quarter"] = date.dt.quarter
    df["days_to_quarter_end"] = (quarter_end - date).dt.days
    df["day_of_week"] = date.dt.dayofweek
    df["day_of_month"] = date.dt.day
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` with every column listed in config.FEATURES."""
    out = df.copy()
    out["log_distance"] = np.log(out["distance"].astype(float))
    out["equipment_code"] = out["equipment"].map(config.EQUIPMENT_CODES)
    if out["equipment_code"].isna().any():
        unknown = sorted(out.loc[out["equipment_code"].isna(), "equipment"].unique())
        raise ValueError(f"Unknown equipment types: {unknown}")
    out["bearing"] = initial_bearing(out["pickup_lat"], out["pickup_lon"], out["delivery_lat"], out["delivery_lon"])
    out = add_calendar_features(out)
    return out
