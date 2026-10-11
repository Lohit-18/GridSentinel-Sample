"""Detectors: physics baseline + machine-learning ensemble."""
from __future__ import annotations

from .physics import PhysicsDetector
from .ml_models import build_base_models, ScoredDetector
from .ensemble import EnsembleDetector
from .pipeline import DetectionPipeline

__all__ = [
    "PhysicsDetector",
    "build_base_models",
    "ScoredDetector",
    "EnsembleDetector",
    "DetectionPipeline",
]
