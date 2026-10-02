"""LightGBM model on log(rate) with additive calendar effects."""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config


@dataclass
class FreightRateModel:
    features: list = field(default_factory=lambda: list(config.FEATURES))
    params: dict = field(default_factory=lambda: dict(config.LGB_PARAMS))
    num_boost_round: int = config.NUM_BOOST_ROUND
    seeds: list = field(default_factory=lambda: list(config.ENSEMBLE_SEEDS))
    boosters: list = field(default_factory=list)

    def _params(self, seed: int) -> dict:
        params = dict(self.params)
        params["seed"] = seed
        params["interaction_constraints"] = [
            [self.features.index(f) for f in group] for group in config.INTERACTION_GROUPS
        ]
        return params

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FreightRateModel":
        target = np.log(np.asarray(y, dtype=float))
        self.boosters = [
            lgb.train(self._params(seed), lgb.Dataset(X[self.features], target), self.num_boost_round)
            for seed in self.seeds
        ]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.boosters:
            raise RuntimeError("Model is not fitted")
        log_pred = np.mean([b.predict(X[self.features]) for b in self.boosters], axis=0)
        return np.exp(log_pred)

    def feature_importance(self) -> pd.Series:
        gain = np.mean([b.feature_importance("gain") for b in self.boosters], axis=0)
        return pd.Series(gain / gain.sum(), index=self.features).sort_values(ascending=False)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: Path) -> "FreightRateModel":
        with open(path, "rb") as fh:
            return pickle.load(fh)
