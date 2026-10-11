"""Central configuration and reproducibility helpers."""
from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np


# --------------------------------------------------------------------------- #
# Global reproducibility
# --------------------------------------------------------------------------- #
GLOBAL_SEED = 1337


def set_global_seed(seed: int = GLOBAL_SEED) -> None:
    """Seed every RNG GridSentinel may touch, for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)


@dataclass
class SimConfig:
    """Parameters for the grid time-series simulation."""

    system: str = "ieee14"          # ieee9 | ieee14 | synth:<n>
    timesteps: int = 2880           # e.g. 2 days at 1-minute resolution
    resolution_min: int = 1         # minutes per step
    meter_noise_std: float = 0.008  # measurement Gaussian noise (p.u.)
    base_load_jitter: float = 0.04  # random walk amplitude on the load profile
    seed: int = GLOBAL_SEED

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AttackConfig:
    """Parameters for the synthetic attack campaign."""

    attack_fraction: float = 0.18          # fraction of timesteps that are attacked
    min_duration: int = 3                  # min consecutive attacked steps
    max_duration: int = 25                 # max consecutive attacked steps
    # relative mix of attack families (auto-normalised)
    mix: dict[str, float] = field(default_factory=lambda: {
        "naive_fdi": 1.0,
        "stealth_fdi": 1.2,
        "ramp_fdi": 0.8,
        "replay": 0.7,
        "goose_spoof": 0.9,
        "goose_flood": 0.5,
    })
    stealth_strength: float = 0.25         # magnitude of residue-preserving shift
    naive_strength: float = 0.12           # magnitude of naive measurement tamper
    seed: int = GLOBAL_SEED + 7

    def normalised_mix(self) -> dict[str, float]:
        total = sum(self.mix.values()) or 1.0
        return {k: v / total for k, v in self.mix.items()}


@dataclass
class DetectConfig:
    """Parameters for feature extraction and the detection ensemble."""

    window: int = 12                 # temporal context window (steps)
    chi2_alpha: float = 0.99         # confidence of the physics bad-data test
    test_size: float = 0.3
    rf_estimators: int = 300
    gb_estimators: int = 200
    iso_estimators: int = 200
    mlp_hidden: tuple[int, ...] = (64, 32)
    seed: int = GLOBAL_SEED + 13


DEFAULT_SIM = SimConfig()
DEFAULT_ATTACK = AttackConfig()
DEFAULT_DETECT = DetectConfig()
