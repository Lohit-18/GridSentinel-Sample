"""False data injection in the *measurement space* (synthetic).

Three families, each returning a tampered copy of the measurement vector ``z``:

- :func:`naive_fdi`   — arbitrary tamper of a few meters. Leaves the column
  space of ``H``, so the chi-square residual grows → the classical detector
  catches it.
- :func:`stealth_fdi` — the residue-preserving attack of Liu, Ning & Reiter
  (2009): the injection ``a = H c`` is built *inside* the column space of ``H``,
  so the WLS residual is unchanged and the chi-square detector is blind. It
  still shifts the estimated state, which is how a temporal/ML detector catches
  it.
- :func:`ramp_fdi`    — a slow bias that grows across the attack window, meant
  to stay under simple rate-of-change alarms.
"""
from __future__ import annotations

import numpy as np

from ..grid.state_estimation import MeasurementModel


def naive_fdi(z: np.ndarray, *, strength: float, rng: np.random.Generator,
              k_meas: int | None = None) -> np.ndarray:
    """Tamper a random subset of meters with an out-of-subspace perturbation."""
    z = z.copy()
    m = z.size
    k = k_meas or max(2, m // 6)
    idx = rng.choice(m, size=min(k, m), replace=False)
    sign = rng.choice([-1.0, 1.0], size=idx.size)
    z[idx] += sign * strength * (np.abs(z[idx]) + 0.15)
    return z


def stealth_fdi(z: np.ndarray, *, model: MeasurementModel, strength: float,
                rng: np.random.Generator, k_states: int | None = None
                ) -> tuple[np.ndarray, np.ndarray]:
    """Residue-preserving attack ``a = H c``.

    Returns the tampered measurements and the induced state shift ``c`` (the
    ground-truth bias the estimator will silently absorb).
    """
    n = model.n_state
    k = k_states or max(1, n // 4)
    c = np.zeros(n)
    tgt = rng.choice(n, size=min(k, n), replace=False)
    c[tgt] = rng.uniform(-1.0, 1.0, size=tgt.size) * strength
    a = model.H @ c
    return z + a, c


def ramp_fdi(z: np.ndarray, *, strength: float, progress: float,
             rng: np.random.Generator, meas_idx: np.ndarray | None = None
             ) -> np.ndarray:
    """A linearly growing bias (``progress`` in [0, 1]) on a fixed meter subset."""
    z = z.copy()
    if meas_idx is None:
        k = max(2, z.size // 8)
        meas_idx = rng.choice(z.size, size=k, replace=False)
    z[meas_idx] += strength * progress * (np.abs(z[meas_idx]) + 0.2)
    return z
