"""DC power flow, the linear measurement model and WLS state estimation.

The *measurement model* stacks every line-flow and every non-slack bus
injection into a vector ``z = H @ theta + noise``, where ``theta`` are the bus
voltage angles (the system state).  Weighted-least-squares recovers an estimate
of ``theta`` and a residual ``r = z - H @ theta_hat``.  The weighted sum of
squared residuals ``J = r' W r`` follows a chi-square distribution under the
"no bad data" hypothesis — this is the classical SCADA bad-data detector that
stealthy FDI attacks are built to defeat.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .ieee_systems import GridSystem


# --------------------------------------------------------------------------- #
# DC power flow
# --------------------------------------------------------------------------- #
def dc_power_flow(system: GridSystem, p_inject: np.ndarray) -> np.ndarray:
    """Solve the DC power flow ``B_red @ theta = p_red`` for bus angles.

    Parameters
    ----------
    p_inject : net active power injection per bus (p.u.); should sum to ~0.
    """
    B = system.susceptance_matrix()
    ns = system.non_slack()
    B_red = B[np.ix_(ns, ns)]
    p_red = p_inject[ns]
    theta = np.zeros(system.n_bus)
    theta[ns] = np.linalg.solve(B_red, p_red)
    return theta


# --------------------------------------------------------------------------- #
# Measurement model
# --------------------------------------------------------------------------- #
@dataclass
class MeasurementModel:
    """Linear DC measurement model ``z = H @ x`` (x = non-slack angles)."""

    H: np.ndarray                 # (m, n_state)
    labels: list[str]             # human-readable name per measurement row
    sigma: np.ndarray             # per-measurement std dev
    system: GridSystem

    @property
    def m(self) -> int:
        return self.H.shape[0]

    @property
    def n_state(self) -> int:
        return self.H.shape[1]

    def W(self) -> np.ndarray:
        """Diagonal weight matrix (inverse measurement variance)."""
        return np.diag(1.0 / self.sigma**2)

    def measure(self, theta: np.ndarray, noise_std: float | None = None,
                rng: np.random.Generator | None = None) -> np.ndarray:
        """Project a true state onto measurements and add Gaussian meter noise."""
        x = theta[self.system.non_slack()]
        z = self.H @ x
        if noise_std is not None:
            rng = rng or np.random.default_rng()
            z = z + rng.normal(0.0, noise_std, size=self.m)
        return z


def build_measurement_model(system: GridSystem,
                            sigma: float = 0.008) -> MeasurementModel:
    """Build a fully-observed DC measurement model.

    Rows: every branch active-power flow, then every non-slack bus injection.
    This gives strong measurement redundancy (``m >> n_state``), which is what
    makes bad-data detection possible in the first place.
    """
    ns = system.non_slack()
    col = {bus: k for k, bus in enumerate(ns)}      # bus -> state column
    rows: list[np.ndarray] = []
    labels: list[str] = []

    # line-flow measurements
    for i, br in enumerate(system.branches):
        row = np.zeros(len(ns))
        if br.f in col:
            row[col[br.f]] += br.b
        if br.t in col:
            row[col[br.t]] -= br.b
        rows.append(row)
        labels.append(f"Pflow[{br.f + 1}->{br.t + 1}]")

    # bus-injection measurements (skip slack)
    B = system.susceptance_matrix()
    for bus in ns:
        row = np.zeros(len(ns))
        for other in range(system.n_bus):
            if other == bus:
                continue
            b_ij = -B[bus, other]
            if b_ij == 0:
                continue
            if bus in col:
                row[col[bus]] += b_ij
            if other in col:
                row[col[other]] -= b_ij
        rows.append(row)
        labels.append(f"Pinj[{bus + 1}]")

    H = np.vstack(rows)
    sig = np.full(H.shape[0], sigma)
    return MeasurementModel(H=H, labels=labels, sigma=sig, system=system)


# --------------------------------------------------------------------------- #
# WLS state estimation + bad-data detection
# --------------------------------------------------------------------------- #
@dataclass
class StateEstimationResult:
    """Outcome of one WLS state-estimation solve."""

    x_hat: np.ndarray              # estimated non-slack angles
    residual: np.ndarray           # z - H x_hat
    normalised_residual: np.ndarray
    j_statistic: float             # r' W r
    threshold: float               # chi-square critical value
    alarm: bool                    # True => classical detector fires
    dof: int

    @property
    def max_normalised_residual(self) -> float:
        return float(np.max(np.abs(self.normalised_residual)))


class DCStateEstimator:
    """Weighted-least-squares DC state estimator with a chi-square detector."""

    def __init__(self, model: MeasurementModel, alpha: float = 0.99):
        self.model = model
        self.alpha = alpha
        H, W = model.H, model.W()
        self.gain = H.T @ W @ H                      # gain matrix G = H'WH
        self.gain_inv = np.linalg.inv(self.gain)
        self.hat = H @ self.gain_inv @ H.T @ W       # hat matrix K
        # residual sensitivity matrix  Omega = S * R  (R = sigma^2 I)
        S = np.eye(model.m) - self.hat
        self.omega = S @ np.diag(model.sigma**2)
        self._omega_diag = np.clip(np.diag(self.omega), 1e-12, None)
        self.dof = max(model.m - model.n_state, 1)
        self.threshold = float(stats.chi2.ppf(alpha, self.dof))

    def estimate(self, z: np.ndarray) -> StateEstimationResult:
        H, W = self.model.H, self.model.W()
        x_hat = self.gain_inv @ H.T @ W @ z
        r = z - H @ x_hat
        j = float(r.T @ W @ r)
        r_norm = r / np.sqrt(self._omega_diag)
        return StateEstimationResult(
            x_hat=x_hat,
            residual=r,
            normalised_residual=r_norm,
            j_statistic=j,
            threshold=self.threshold,
            alarm=bool(j > self.threshold),
            dof=self.dof,
        )
