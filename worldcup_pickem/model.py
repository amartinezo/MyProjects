"""The ML core: a gradient-boosted model that predicts a team's expected goals.

We train ONE regressor on team-perspective rows (each match contributes two
rows: home-vs-away and away-vs-home). Using ``loss="poisson"`` makes the model
output a non-negative expected goal count - exactly the rate parameter a Poisson
score grid needs. This is the "ML predicts the goal rates, Poisson turns rates
into a score distribution" hybrid.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import config
from features import FEATURE_COLS


class GoalsModel:
    def __init__(self, **kw):
        params = dict(
            loss="poisson",
            max_depth=4,
            learning_rate=0.05,
            max_iter=400,
            min_samples_leaf=25,
            l2_regularization=1.0,
            random_state=42,
        )
        params.update(kw)
        self.reg = HistGradientBoostingRegressor(**params)
        self.trained = False

    def fit(self, feat: pd.DataFrame) -> "GoalsModel":
        X = feat[FEATURE_COLS]
        y = feat["goals"].astype(float)
        self.reg.fit(X, y)
        self.trained = True
        return self

    def predict_lambda(self, X: pd.DataFrame) -> np.ndarray:
        """Expected goals for each row (clipped to a sensible positive range)."""
        pred = self.reg.predict(X[FEATURE_COLS])
        return np.clip(pred, 0.05, 8.0)

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path) if path else config.MODEL_DIR / "goals_model.joblib"
        joblib.dump(self.reg, path)
        return path

    @classmethod
    def load(cls, path: str | Path | None = None) -> "GoalsModel":
        path = Path(path) if path else config.MODEL_DIR / "goals_model.joblib"
        obj = cls()
        obj.reg = joblib.load(path)
        obj.trained = True
        return obj
