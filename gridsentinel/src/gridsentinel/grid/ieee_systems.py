"""IEEE benchmark power systems and a procedural synthetic generator.

A :class:`GridSystem` is a lightweight, DC-model description of a transmission
network: buses, branches (with series reactance) and per-bus load / generation
participation. Branch reactances for the IEEE 9- and 14-bus systems are the
standard textbook values; the synthetic generator builds an arbitrarily large
ring-plus-radial network for scalability demonstrations.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Branch:
    """A transmission line / transformer in the DC model."""

    f: int          # from-bus (0-indexed)
    t: int          # to-bus   (0-indexed)
    x: float        # series reactance (p.u.)

    @property
    def b(self) -> float:
        """Series susceptance used by the DC model."""
        return 1.0 / self.x


@dataclass
class GridSystem:
    """A DC-model power system."""

    name: str
    n_bus: int
    branches: list[Branch]
    slack: int = 0                                   # reference bus (0-indexed)
    load_share: np.ndarray = field(default_factory=lambda: np.array([]))
    gen_buses: tuple[int, ...] = (0,)
    base_mva: float = 100.0

    # ---- derived topology ------------------------------------------------- #
    @property
    def n_branch(self) -> int:
        return len(self.branches)

    def susceptance_matrix(self) -> np.ndarray:
        """Full nodal susceptance matrix ``B`` (n_bus x n_bus)."""
        B = np.zeros((self.n_bus, self.n_bus))
        for br in self.branches:
            B[br.f, br.f] += br.b
            B[br.t, br.t] += br.b
            B[br.f, br.t] -= br.b
            B[br.t, br.f] -= br.b
        return B

    def non_slack(self) -> list[int]:
        return [i for i in range(self.n_bus) if i != self.slack]

    def neighbours(self) -> dict[int, list[int]]:
        """Adjacency map used by the spatial-consistency features."""
        adj: dict[int, list[int]] = {i: [] for i in range(self.n_bus)}
        for br in self.branches:
            adj[br.f].append(br.t)
            adj[br.t].append(br.f)
        return adj

    def default_load_share(self) -> np.ndarray:
        """Return (or lazily build) a per-bus load participation vector."""
        if self.load_share.size == self.n_bus:
            return self.load_share
        rng = np.random.default_rng(abs(hash(self.name)) % (2**32))
        share = rng.uniform(0.2, 1.0, size=self.n_bus)
        for g in self.gen_buses:
            share[g] *= 0.15                         # generator buses carry little load
        share /= share.sum()
        self.load_share = share
        return share


# --------------------------------------------------------------------------- #
# Standard IEEE systems (reactances in p.u. on a 100 MVA base)
# --------------------------------------------------------------------------- #
_IEEE9 = [
    (1, 4, 0.0576), (4, 5, 0.0920), (5, 6, 0.1700), (3, 6, 0.0586),
    (6, 7, 0.1008), (7, 8, 0.0720), (8, 2, 0.0625), (8, 9, 0.1610),
    (9, 4, 0.0850),
]

_IEEE14 = [
    (1, 2, 0.05917), (1, 5, 0.22304), (2, 3, 0.19797), (2, 4, 0.17632),
    (2, 5, 0.17388), (3, 4, 0.17103), (4, 5, 0.04211), (4, 7, 0.20912),
    (4, 9, 0.55618), (5, 6, 0.25202), (6, 11, 0.19890), (6, 12, 0.25581),
    (6, 13, 0.13027), (7, 8, 0.17615), (7, 9, 0.11001), (9, 10, 0.08450),
    (9, 14, 0.27038), (10, 11, 0.19207), (12, 13, 0.19988), (13, 14, 0.34802),
]


def _branches_from_pairs(pairs: list[tuple[int, int, float]]) -> list[Branch]:
    # convert 1-indexed textbook buses to 0-indexed internal buses
    return [Branch(f - 1, t - 1, x) for (f, t, x) in pairs]


def _synthetic(n_bus: int, seed: int = 12345) -> GridSystem:
    """Build a connected ring + random-chord network of ``n_bus`` buses."""
    rng = np.random.default_rng(seed)
    branches: list[tuple[int, int, float]] = []
    for i in range(n_bus):                           # ring guarantees connectivity
        j = (i + 1) % n_bus
        branches.append((i + 1, j + 1, float(rng.uniform(0.04, 0.30))))
    n_chord = max(1, n_bus // 3)                     # extra chords for redundancy
    for _ in range(n_chord):
        a, b = rng.integers(1, n_bus + 1, size=2)
        if a != b:
            branches.append((int(a), int(b), float(rng.uniform(0.05, 0.4))))
    return GridSystem(
        name=f"synth{n_bus}",
        n_bus=n_bus,
        branches=_branches_from_pairs(branches),
        slack=0,
        gen_buses=(0, n_bus // 2),
        base_mva=100.0,
    )


_REGISTRY = {
    "ieee9": lambda: GridSystem("ieee9", 9, _branches_from_pairs(_IEEE9),
                                slack=0, gen_buses=(0, 1, 2)),
    "ieee14": lambda: GridSystem("ieee14", 14, _branches_from_pairs(_IEEE14),
                                 slack=0, gen_buses=(0, 1, 2, 5, 7)),
}


def available_systems() -> list[str]:
    """Names of the built-in systems (plus the ``synth:<n>`` family)."""
    return sorted(_REGISTRY) + ["synth:<n>"]


def load_system(name: str = "ieee14") -> GridSystem:
    """Load a grid system by name.

    ``ieee9`` / ``ieee14`` return the standard benchmarks; ``synth:<n>``
    (e.g. ``synth:30``) builds a synthetic network with ``n`` buses.
    """
    name = name.strip().lower()
    if name.startswith("synth:"):
        n = int(name.split(":", 1)[1])
        if n < 4:
            raise ValueError("synthetic systems need at least 4 buses")
        return _synthetic(n)
    if name not in _REGISTRY:
        raise KeyError(f"unknown system {name!r}; try {available_systems()}")
    sys = _REGISTRY[name]()
    sys.default_load_share()
    return sys
