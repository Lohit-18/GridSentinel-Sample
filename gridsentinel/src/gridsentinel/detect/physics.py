"""The classical SCADA bad-data detector, wrapped as a baseline.

It fires when the weighted sum of squared residuals exceeds the chi-square
critical value.  It is fast, interpretable and standard in industry — and, by
construction, blind to the residue-preserving stealth attack.  GridSentinel's
whole value proposition is measured *against* this baseline.
"""
from __future__ import annotations

import numpy as np

from ..features.extractor import FeatureMatrix


class PhysicsDetector:
    """Chi-square residual detector operating on the precomputed J statistic."""

    name = "physics_chi2"

    def __init__(self, threshold: float | None = None):
        self.threshold = threshold

    def fit(self, fm: FeatureMatrix) -> "PhysicsDetector":
        # the threshold is a statistical critical value, not learned from data
        self.threshold = fm.threshold if self.threshold is None else self.threshold
        return self

    def score(self, fm: FeatureMatrix) -> np.ndarray:
        """Normalised exceedance of the chi-square threshold, squashed to [0, 1)."""
        ratio = fm.j_stat / (self.threshold or fm.threshold)
        return 1.0 - 1.0 / (1.0 + np.maximum(ratio - 1.0, 0.0))

    def predict(self, fm: FeatureMatrix) -> np.ndarray:
        return (fm.j_stat > (self.threshold or fm.threshold)).astype(int)
