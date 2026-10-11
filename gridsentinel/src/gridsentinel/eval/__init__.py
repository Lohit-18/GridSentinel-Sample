"""Evaluation: metrics, detector comparison and experiment orchestration."""
from __future__ import annotations

from .metrics import (
    binary_metrics,
    per_attack_recall,
    detection_latency,
    roc_curve_points,
    pr_curve_points,
)
from .experiments import (
    ExperimentResult,
    run_experiment,
    run_scaling_study,
)

__all__ = [
    "binary_metrics",
    "per_attack_recall",
    "detection_latency",
    "roc_curve_points",
    "pr_curve_points",
    "ExperimentResult",
    "run_experiment",
    "run_scaling_study",
]
