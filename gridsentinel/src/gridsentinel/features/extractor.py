"""Turn a time-ordered campaign into a per-timestep feature matrix.

Features span four physical/logical domains so that *no single attack family*
can evade every one of them:

- **estimation**  : chi-square statistic and residual shape (catches naive FDI).
- **state**       : how far / how fast the estimated state moved (catches the
  stealthy residue-preserving FDI the chi-square test is blind to).
- **measurement** : windowed z-scores and step changes of the raw meters.
- **spatial**     : Kirchhoff and neighbour-consistency checks (localises the
  anomaly against the broader grid context).
- **protocol**    : GOOSE stNum / sqNum / timing anomalies (catches replay and
  floods that never touch the analog values enough to disturb the physics).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from ..attacks.scenarios import AttackCampaign
from ..grid.state_estimation import DCStateEstimator

FEATURE_NAMES = [
    # estimation
    "j_statistic", "j_ratio", "max_norm_residual", "mean_abs_residual", "std_residual",
    # state
    "xhat_norm", "state_step_l2", "state_window_dev",
    # measurement
    "dz_l2", "dz_max", "z_window_max", "z_window_mean",
    # spatial
    "spatial_max_dev", "spatial_mean_dev", "kirchhoff_max",
    # protocol
    "st_delta", "sq_delta", "dt_ms", "dt_zscore", "stale", "flood_flag",
]

FEATURE_DOMAIN = {
    **{n: "estimation" for n in FEATURE_NAMES[0:5]},
    **{n: "state" for n in FEATURE_NAMES[5:8]},
    **{n: "measurement" for n in FEATURE_NAMES[8:12]},
    **{n: "spatial" for n in FEATURE_NAMES[12:15]},
    **{n: "protocol" for n in FEATURE_NAMES[15:]},
}


@dataclass
class FeatureMatrix:
    """Extracted features and aligned labels."""

    X: np.ndarray                 # (T, n_feat)
    y: np.ndarray                 # (T,)
    atype: np.ndarray             # (T,)
    j_stat: np.ndarray            # (T,) chi-square statistic (physics baseline)
    threshold: float
    names: list[str]

    @property
    def T(self) -> int:
        return self.X.shape[0]


class FeatureExtractor:
    """Compute :data:`FEATURE_NAMES` for every timestep of a campaign."""

    _flow_re = re.compile(r"Pflow\[(\d+)->(\d+)\]")
    _inj_re = re.compile(r"Pinj\[(\d+)\]")

    def __init__(self, estimator: DCStateEstimator, window: int = 12):
        self.est = estimator
        self.window = window
        self.model = estimator.model
        self.system = self.model.system
        self._build_topology_maps()

    def _build_topology_maps(self) -> None:
        self.flow_rows: list[tuple[int, int, int]] = []     # (row, from, to) 0-indexed
        self.inj_row: dict[int, int] = {}
        for r, lab in enumerate(self.model.labels):
            mf = self._flow_re.match(lab)
            mi = self._inj_re.match(lab)
            if mf:
                self.flow_rows.append((r, int(mf.group(1)) - 1, int(mf.group(2)) - 1))
            elif mi:
                self.inj_row[int(mi.group(1)) - 1] = r
        self.neighbours = self.system.neighbours()
        self.ns = self.system.non_slack()
        self.col = {b: k for k, b in enumerate(self.ns)}

    def _kirchhoff_max(self, z: np.ndarray) -> float:
        """Max |measured injection − net measured flow| across buses."""
        worst = 0.0
        for bus, ir in self.inj_row.items():
            net = 0.0
            for r, f, t in self.flow_rows:
                if f == bus:
                    net += z[r]
                elif t == bus:
                    net -= z[r]
            worst = max(worst, abs(z[ir] - net))
        return worst

    def _spatial_dev(self, theta_full: np.ndarray) -> tuple[float, float]:
        devs = []
        for bus, nbrs in self.neighbours.items():
            if not nbrs:
                continue
            devs.append(abs(theta_full[bus] - np.mean(theta_full[nbrs])))
        devs = np.array(devs) if devs else np.array([0.0])
        return float(devs.max()), float(devs.mean())

    def extract(self, campaign: AttackCampaign) -> FeatureMatrix:
        Z = campaign.Z_obs
        T, m = Z.shape
        W = self.window
        n_state = self.model.n_state

        # precompute estimator outputs
        x_hat = np.zeros((T, n_state))
        j_stat = np.zeros(T)
        max_nr = np.zeros(T)
        mean_res = np.zeros(T)
        std_res = np.zeros(T)
        theta_full = np.zeros((T, self.system.n_bus))
        for t in range(T):
            res = self.est.estimate(Z[t])
            x_hat[t] = res.x_hat
            j_stat[t] = res.j_statistic
            max_nr[t] = res.max_normalised_residual
            mean_res[t] = float(np.mean(np.abs(res.residual)))
            std_res[t] = float(np.std(res.residual))
            theta_full[t, self.ns] = res.x_hat

        X = np.zeros((T, len(FEATURE_NAMES)))
        thr = self.est.threshold
        for t in range(T):
            lo = max(0, t - W)
            z = Z[t]
            # estimation
            X[t, 0] = j_stat[t]
            X[t, 1] = j_stat[t] / thr
            X[t, 2] = max_nr[t]
            X[t, 3] = mean_res[t]
            X[t, 4] = std_res[t]
            # state
            X[t, 5] = float(np.linalg.norm(x_hat[t]))
            X[t, 6] = float(np.linalg.norm(x_hat[t] - x_hat[t - 1])) if t > 0 else 0.0
            if t > lo:
                base = x_hat[lo:t]
                dev = np.linalg.norm(x_hat[t] - base.mean(0))
                X[t, 7] = dev / (base.std(0).mean() + 1e-6)
            # measurement
            X[t, 8] = float(np.linalg.norm(z - Z[t - 1])) if t > 0 else 0.0
            X[t, 9] = float(np.max(np.abs(z - Z[t - 1]))) if t > 0 else 0.0
            if t > lo:
                win = Z[lo:t]
                mu, sd = win.mean(0), win.std(0) + 1e-6
                zs = np.abs(z - mu) / sd
                X[t, 10] = float(zs.max())
                X[t, 11] = float(zs.mean())
            # spatial
            smax, smean = self._spatial_dev(theta_full[t])
            X[t, 12] = smax
            X[t, 13] = smean
            X[t, 14] = self._kirchhoff_max(z)
            # protocol
            X[t, 15] = float(campaign.st_num[t] - campaign.st_num[t - 1]) if t > 0 else 0.0
            X[t, 16] = float(campaign.sq_num[t] - campaign.sq_num[t - 1]) if t > 0 else 0.0
            X[t, 17] = campaign.dt_ms[t]
            if t > lo:
                d = campaign.dt_ms[lo:t]
                X[t, 18] = abs(campaign.dt_ms[t] - d.mean()) / (d.std() + 1e-6)
            X[t, 19] = float(campaign.stale[t])
            X[t, 20] = 1.0 if X[t, 16] > 10 else 0.0

        return FeatureMatrix(X=X, y=campaign.y.copy(), atype=campaign.atype.copy(),
                             j_stat=j_stat, threshold=thr, names=list(FEATURE_NAMES))
