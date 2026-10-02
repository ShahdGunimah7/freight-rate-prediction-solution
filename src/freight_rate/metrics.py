from __future__ import annotations

import numpy as np


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    ss_res = float((err**2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return {
        "n": int(len(y_true)),
        "MAE": float(np.abs(err).mean()),
        "RMSE": float(np.sqrt((err**2).mean())),
        "MAPE_%": float(100 * np.mean(np.abs(err) / y_true)),
        "MedianAPE_%": float(100 * np.median(np.abs(err) / y_true)),
        "R2": 1 - ss_res / ss_tot,
    }
