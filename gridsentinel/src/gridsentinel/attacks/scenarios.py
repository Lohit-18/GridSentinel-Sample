"""Compose individual attacks into a labelled, time-ordered campaign."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import AttackConfig
from ..grid.simulator import SimulationData
from . import fdi, goose_attacks
from .goose_attacks import GooseMeta

ATTACK_TYPES = [
    "normal", "naive_fdi", "stealth_fdi", "ramp_fdi",
    "replay", "goose_spoof", "goose_flood",
]


@dataclass
class AttackCampaign:
    """A clean simulation plus its tampered, labelled counterpart."""

    sim: SimulationData
    Z_obs: np.ndarray            # (T, m) observed measurements (tampered)
    y: np.ndarray                # (T,) 1 = attacked
    atype: np.ndarray            # (T,) attack-type string
    st_num: np.ndarray           # (T,) GOOSE stNum
    sq_num: np.ndarray           # (T,) GOOSE sqNum
    dt_ms: np.ndarray            # (T,) inter-arrival time
    stale: np.ndarray            # (T,) stale-timestamp flag (0/1)
    state_shift: np.ndarray      # (T,) injected state-shift L2 norm (ground truth)
    config: AttackConfig

    @property
    def T(self) -> int:
        return self.Z_obs.shape[0]

    def summary(self) -> dict[str, int]:
        vals, counts = np.unique(self.atype, return_counts=True)
        return {str(v): int(c) for v, c in zip(vals, counts)}


def _plan_windows(T: int, cfg: AttackConfig, rng: np.random.Generator
                  ) -> list[tuple[int, int, str]]:
    """Carve out non-overlapping attack windows covering ~attack_fraction of T."""
    mix = cfg.normalised_mix()
    types = list(mix)
    probs = np.array([mix[t] for t in types])
    budget = int(cfg.attack_fraction * T)
    margin = 15                                  # keep the first steps attack-free
    windows: list[tuple[int, int, str]] = []
    used = np.zeros(T, dtype=bool)
    guard = 0
    spent = 0
    while spent < budget and guard < 10 * T:
        guard += 1
        dur = int(rng.integers(cfg.min_duration, cfg.max_duration + 1))
        start = int(rng.integers(margin + 1, max(margin + 2, T - dur - 1)))
        if used[max(0, start - 2):start + dur + 2].any():
            continue
        atk = str(rng.choice(types, p=probs))
        windows.append((start, start + dur, atk))
        used[start:start + dur] = True
        spent += dur
    return sorted(windows)


def generate_campaign(sim: SimulationData,
                      cfg: AttackConfig | None = None) -> AttackCampaign:
    """Build a labelled attack campaign on top of a clean simulation."""
    cfg = cfg or AttackConfig()
    rng = np.random.default_rng(cfg.seed)
    T, m = sim.Z.shape
    Z_obs = sim.Z.copy()
    y = np.zeros(T, dtype=int)
    atype = np.array(["normal"] * T, dtype=object)
    st = np.zeros(T, dtype=int)
    sq = np.zeros(T, dtype=int)
    dt = np.zeros(T, dtype=float)
    stale = np.zeros(T, dtype=int)
    shift = np.zeros(T, dtype=float)

    windows = _plan_windows(T, cfg, rng)
    win_at = {}
    for start, end, atk in windows:
        for t in range(start, end):
            win_at[t] = (atk, start, end)

    cur_st, cur_sq = 1, 0
    last_good_z = sim.Z[0].copy()

    for t in range(T):
        # ---- normal control-plane evolution --------------------------------
        if rng.random() < 0.15:            # a monitored value "changed"
            cur_st += 1
            cur_sq = 0
        else:
            cur_sq += 1
        meta = GooseMeta(st_num=cur_st, sq_num=cur_sq,
                         dt_ms=float(rng.normal(1000, 25)), stale=False)

        if t in win_at:
            atk, start, end = win_at[t]
            y[t] = 1
            atype[t] = atk
            if atk == "naive_fdi":
                Z_obs[t] = fdi.naive_fdi(sim.Z[t], strength=cfg.naive_strength, rng=rng)
            elif atk == "stealth_fdi":
                z_new, c = fdi.stealth_fdi(sim.Z[t], model=sim.model,
                                           strength=cfg.stealth_strength, rng=rng)
                Z_obs[t] = z_new
                shift[t] = float(np.linalg.norm(c))
            elif atk == "ramp_fdi":
                prog = (t - start + 1) / max(1, end - start)
                Z_obs[t] = fdi.ramp_fdi(sim.Z[t], strength=cfg.naive_strength * 3,
                                        progress=prog, rng=rng)
            elif atk == "replay":
                Z_obs[t], meta = goose_attacks.goose_replay(
                    last_good_z, prev_st=cur_st, rng=rng)
            elif atk == "goose_spoof":
                Z_obs[t], meta = goose_attacks.goose_spoof(
                    sim.Z[t], strength=cfg.stealth_strength, rng=rng, prev_st=cur_st)
                cur_st = meta.st_num
            elif atk == "goose_flood":
                Z_obs[t], meta = goose_attacks.goose_flood(
                    sim.Z[t], prev_st=cur_st, prev_sq=cur_sq, rng=rng)
                cur_st, cur_sq = meta.st_num, meta.sq_num
        else:
            last_good_z = sim.Z[t].copy()

        st[t], sq[t], dt[t] = meta.st_num, meta.sq_num, meta.dt_ms
        stale[t] = int(meta.stale)

    return AttackCampaign(
        sim=sim, Z_obs=Z_obs, y=y, atype=atype, st_num=st, sq_num=sq,
        dt_ms=dt, stale=stale, state_shift=shift, config=cfg,
    )
