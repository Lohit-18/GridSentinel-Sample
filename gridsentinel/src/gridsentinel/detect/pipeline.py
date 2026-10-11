"""End-to-end detection pipeline: campaign -> features -> ensemble verdicts."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..attacks.scenarios import AttackCampaign
from ..config import DetectConfig
from ..features.extractor import FeatureExtractor, FeatureMatrix
from ..grid.state_estimation import DCStateEstimator
from .ensemble import EnsembleDetector
from .physics import PhysicsDetector


@dataclass
class DetectionResult:
    """Verdicts and scores for one campaign."""

    y_true: np.ndarray
    atype: np.ndarray
    ensemble_score: np.ndarray
    ensemble_pred: np.ndarray
    physics_pred: np.ndarray
    physics_score: np.ndarray
    base_scores: dict[str, np.ndarray]
    j_stat: np.ndarray
    threshold: float


class DetectionPipeline:
    """Owns the estimator, feature extractor and fitted ensemble."""

    def __init__(self, model, cfg: DetectConfig | None = None, alpha: float = 0.99):
        self.cfg = cfg or DetectConfig()
        self.estimator = DCStateEstimator(model, alpha=alpha)
        self.extractor = FeatureExtractor(self.estimator, window=self.cfg.window)
        self.ensemble = EnsembleDetector(self.cfg)
        self.physics = PhysicsDetector()
        self.fitted = False

    # ------------------------------------------------------------------ #
    def _choose_threshold(self, scores: np.ndarray, y: np.ndarray) -> float:
        """Pick the probability cut that maximises F1 on the training split."""
        best_t, best_f1 = 0.5, -1.0
        for t in np.linspace(0.05, 0.95, 37):
            pred = (scores >= t).astype(int)
            tp = int(((pred == 1) & (y == 1)).sum())
            fp = int(((pred == 1) & (y == 0)).sum())
            fn = int(((pred == 0) & (y == 1)).sum())
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        return best_t

    def fit(self, train: AttackCampaign) -> "DetectionPipeline":
        fm = self.extractor.extract(train)
        jr = fm.j_stat / fm.threshold
        self.ensemble.fit(fm.X, fm.y, jr)
        self.physics.fit(fm)
        train_scores = self.ensemble.score(fm.X, jr)
        self.ensemble.operating_threshold = self._choose_threshold(train_scores, fm.y)
        self.fitted = True
        return self

    # ------------------------------------------------------------------ #
    def features(self, campaign: AttackCampaign) -> FeatureMatrix:
        return self.extractor.extract(campaign)

    def predict(self, campaign: AttackCampaign) -> DetectionResult:
        if not self.fitted:
            raise RuntimeError("pipeline must be fitted before predicting")
        fm = self.extractor.extract(campaign)
        jr = fm.j_stat / fm.threshold
        score = self.ensemble.score(fm.X, jr)
        return DetectionResult(
            y_true=fm.y,
            atype=fm.atype,
            ensemble_score=score,
            ensemble_pred=(score >= self.ensemble.operating_threshold).astype(int),
            physics_pred=self.physics.predict(fm),
            physics_score=self.physics.score(fm),
            base_scores=self.ensemble.base_scores(fm.X),
            j_stat=fm.j_stat,
            threshold=fm.threshold,
        )

    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        import joblib
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "DetectionPipeline":
        import joblib
        return joblib.load(path)
