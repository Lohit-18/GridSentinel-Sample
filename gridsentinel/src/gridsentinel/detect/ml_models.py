"""Base machine-learning detectors with a common scoring interface."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    IsolationForest,
)
from sklearn.neural_network import MLPClassifier

from ..config import DetectConfig


@dataclass
class ScoredDetector:
    """Wraps a fitted estimator to expose a uniform ``score`` in [0, 1]."""

    name: str
    model: object
    kind: str                     # "supervised" | "anomaly"

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.kind == "supervised":
            return self.model.predict_proba(X)[:, 1]
        # IsolationForest: higher score_samples = more normal -> invert & squash
        raw = -self.model.score_samples(X)
        lo, hi = np.percentile(raw, 1), np.percentile(raw, 99)
        return np.clip((raw - lo) / (hi - lo + 1e-9), 0.0, 1.0)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.score(X) >= threshold).astype(int)


def build_base_models(cfg: DetectConfig | None = None) -> dict[str, object]:
    """Instantiate the (unfitted) base learners used by the ensemble."""
    cfg = cfg or DetectConfig()
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=cfg.rf_estimators, max_depth=None, n_jobs=-1,
            class_weight="balanced_subsample", random_state=cfg.seed,
        ),
        "gradient_boost": GradientBoostingClassifier(
            n_estimators=cfg.gb_estimators, max_depth=3,
            learning_rate=0.08, random_state=cfg.seed,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=cfg.mlp_hidden, activation="relu",
            alpha=1e-3, max_iter=400, random_state=cfg.seed,
        ),
        "isolation_forest": IsolationForest(
            n_estimators=cfg.iso_estimators, contamination="auto",
            n_jobs=-1, random_state=cfg.seed,
        ),
    }
