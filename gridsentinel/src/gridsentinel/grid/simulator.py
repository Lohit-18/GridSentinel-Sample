"""Time-series grid simulator.

Produces a realistic stream of SCADA measurements by driving the DC power flow
with a daily load curve plus a bounded random walk.  The output is what a
substation's state estimator would normally see — the "normal ebb and flow" the
detectors learn from.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import SimConfig
from .ieee_systems import GridSystem, load_system
from .state_estimation import (
    MeasurementModel,
    build_measurement_model,
    dc_power_flow,
)


@dataclass
class SimulationData:
    """A full clean simulation run."""

    system: GridSystem
    model: MeasurementModel
    Z: np.ndarray            # (T, m) clean measurements
    THETA: np.ndarray        # (T, n_bus) true states
    P_INJ: np.ndarray        # (T, n_bus) net injections
    load_total: np.ndarray   # (T,) total system load (p.u.)
    minutes: np.ndarray      # (T,) minute-of-run timestamps
    config: SimConfig

    @property
    def T(self) -> int:
        return self.Z.shape[0]


def _daily_profile(t_min: np.ndarray) -> np.ndarray:
    """A two-hump daily demand curve (morning + evening peaks), in [0.6, 1.25]."""
    phase = 2 * np.pi * (t_min % 1440) / 1440.0
    base = 0.92 + 0.18 * np.sin(phase - np.pi / 2.2)        # broad daily swing
    evening = 0.12 * np.exp(-((phase - 2 * np.pi * 19 / 24) ** 2) / 0.12)
    morning = 0.06 * np.exp(-((phase - 2 * np.pi * 8 / 24) ** 2) / 0.10)
    return np.clip(base + evening + morning, 0.55, 1.3)


def simulate(config: SimConfig | None = None) -> SimulationData:
    """Run a clean (attack-free) simulation."""
    cfg = config or SimConfig()
    rng = np.random.default_rng(cfg.seed)
    system = load_system(cfg.system)
    model = build_measurement_model(system, sigma=cfg.meter_noise_std)

    T = cfg.timesteps
    minutes = np.arange(T) * cfg.resolution_min
    share = system.default_load_share()

    # system-level demand: daily curve * bounded random walk
    profile = _daily_profile(minutes.astype(float))
    walk = np.cumsum(rng.normal(0.0, cfg.base_load_jitter, size=T))
    walk -= walk.mean()
    walk = np.tanh(walk) * 0.12                                   # keep it bounded
    base_total = 2.0 + 0.6 * system.n_bus / 14.0                  # scale with size
    load_total = base_total * profile * (1.0 + walk)

    Z = np.zeros((T, model.m))
    THETA = np.zeros((T, system.n_bus))
    P_INJ = np.zeros((T, system.n_bus))
    gen = np.array(system.gen_buses)

    for t in range(T):
        load = load_total[t] * share
        inject = -load.copy()                                    # consumption is negative
        # generators (non-slack) supply a participation share of total load
        part = rng.uniform(0.8, 1.2, size=len(gen))
        part = part / part.sum()
        for k, g in enumerate(gen):
            if g == system.slack:
                continue
            inject[g] += load_total[t] * part[k] * 0.85
        inject[system.slack] -= inject.sum()                     # slack balances
        theta = dc_power_flow(system, inject)
        THETA[t] = theta
        P_INJ[t] = inject
        Z[t] = model.measure(theta, noise_std=cfg.meter_noise_std, rng=rng)

    return SimulationData(
        system=system, model=model, Z=Z, THETA=THETA, P_INJ=P_INJ,
        load_total=load_total, minutes=minutes, config=cfg,
    )
