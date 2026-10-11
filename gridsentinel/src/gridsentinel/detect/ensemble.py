"""The GridSentinel stacked ensemble.

Three supervised base learners (random forest, gradient boosting, MLP) are
combined with an unsupervised anomaly score (isolation forest) and the physics
chi-square ratio through a logistic-regression meta-learner.  Out-of-fold
predictions feed the meta-learner so the stack does not overfit its own bases.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

from ..config import DetectConfig
from .ml_models import ScoredDetector, build_base_models

_SUPERVISED = ["random_forest", "gradient_boost", "mlp"]


class EnsembleDetector:
    """Fit/score a stacked ensemble over engineered features + physics."""

    name = "gridsentinel_ensemble"

    def __init__(self, cfg: DetectConfig | None = None):
        self.cfg = cfg or DetectConfig()
        self.scaler = StandardScaler()
        self.bases: dict[str, ScoredDetector] = {}
        self.iso: ScoredDetector | None = None
        self.meta = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.meta_cols: list[str] = []
        self.feature_importances_: np.ndarray | None = None
        self.operating_threshold = 0.5

    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, y: np.ndarray, j_ratio: np.ndarray
            ) -> "EnsembleDetector":
        Xs = self.scaler.fit_transform(X)
        raw = build_base_models(self.cfg)
        cv = 3 if (y == 1).sum() >= 30 else 2

        meta_features = []
        self.meta_cols = []
        for name in _SUPERVISED:
            model = raw[name]
            oof = cross_val_predict(model, Xs, y, cv=cv,
                                    method="predict_proba", n_jobs=-1)[:, 1]
            model.fit(Xs, y)                               # refit on all data
            self.bases[name] = ScoredDetector(name, model, "supervised")
            meta_features.append(oof)
            self.meta_cols.append(name)

        iso = raw["isolation_forest"]
        iso.fit(Xs[y == 0] if (y == 0).any() else Xs)      # learn "normal"
        self.iso = ScoredDetector("isolation_forest", iso, "anomaly")
        meta_features.append(self.iso.score(Xs))
        self.meta_cols.append("isolation_forest")

        meta_features.append(np.clip(j_ratio, 0, 50))
        self.meta_cols.append("physics_chi2")

        M = np.column_stack(meta_features)
        self.meta.fit(M, y)
        self.feature_importances_ = self.bases["random_forest"].model.feature_importances_
        return self

    # ------------------------------------------------------------------ #
    def _meta_matrix(self, X: np.ndarray, j_ratio: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        cols = [self.bases[n].score(Xs) for n in _SUPERVISED]
        cols.append(self.iso.score(Xs))
        cols.append(np.clip(j_ratio, 0, 50))
        return np.column_stack(cols)

    def score(self, X: np.ndarray, j_ratio: np.ndarray) -> np.ndarray:
        return self.meta.predict_proba(self._meta_matrix(X, j_ratio))[:, 1]

    def predict(self, X: np.ndarray, j_ratio: np.ndarray,
                threshold: float | None = None) -> np.ndarray:
        thr = self.operating_threshold if threshold is None else threshold
        return (self.score(X, j_ratio) >= thr).astype(int)

    def base_scores(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Per-base-learner scores, for diagnostics and the dashboard."""
        Xs = self.scaler.transform(X)
        out = {n: self.bases[n].score(Xs) for n in _SUPERVISED}
        out["isolation_forest"] = self.iso.score(Xs)
        return out
