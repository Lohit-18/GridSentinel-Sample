"""GOOSE-protocol attacks (synthetic).

These model the publisher-side manipulations an attacker with a foothold on the
substation LAN could attempt.  Each returns a tampered measurement vector plus
a :class:`GooseMeta` describing the *protocol* anomalies (stNum / sqNum /
timing), which the detector consumes as additional features.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GooseMeta:
    """Per-message GOOSE control-field state used for protocol features."""

    st_num: int
    sq_num: int
    dt_ms: float            # inter-arrival time since previous message
    ttl_ms: int = 2000
    stale: bool = False     # timestamp older than expected
    test_flag: bool = False


def goose_spoof(z: np.ndarray, *, strength: float, rng: np.random.Generator,
                prev_st: int, k_entries: int | None = None
                ) -> tuple[np.ndarray, GooseMeta]:
    """Spoof a few analog dataset entries while keeping control fields *valid*.

    This is the nasty one: ``stNum`` increments legitimately (a value changed),
    timing is normal, so every protocol-level check passes. Only the physics or
    a learned model can flag it.
    """
    z = z.copy()
    k = k_entries or max(1, z.size // 8)
    idx = rng.choice(z.size, size=min(k, z.size), replace=False)
    z[idx] = z[idx] + rng.choice([-1.0, 1.0], size=idx.size) * strength * (np.abs(z[idx]) + 0.2)
    meta = GooseMeta(st_num=prev_st + 1, sq_num=0, dt_ms=1000.0, stale=False)
    return z, meta


def goose_replay(z_prev_good: np.ndarray, *, prev_st: int,
                 rng: np.random.Generator) -> tuple[np.ndarray, GooseMeta]:
    """Replay a previously captured (now stale) message.

    The values are real-but-old; the giveaway is the control plane — ``stNum``
    rolls back and the timestamp is stale.
    """
    rollback = int(rng.integers(1, 4))
    meta = GooseMeta(
        st_num=max(1, prev_st - rollback),         # stNum should never decrease
        sq_num=int(rng.integers(40, 120)),
        dt_ms=1000.0,
        stale=True,
    )
    return z_prev_good.copy(), meta


def goose_flood(z: np.ndarray, *, prev_st: int, prev_sq: int,
                rng: np.random.Generator) -> tuple[np.ndarray, GooseMeta]:
    """High-rate spoof flood: a burst of messages with jumping sqNum."""
    z = z.copy()
    # attacker pins a subset of meters to an implausible constant
    k = max(1, z.size // 5)
    idx = rng.choice(z.size, size=min(k, z.size), replace=False)
    z[idx] = rng.uniform(-3.0, 3.0, size=idx.size)
    meta = GooseMeta(
        st_num=prev_st + int(rng.integers(3, 20)),
        sq_num=prev_sq + int(rng.integers(50, 400)),
        dt_ms=float(rng.uniform(0.2, 2.0)),         # sub-millisecond burst
        stale=False,
    )
    return z, meta
