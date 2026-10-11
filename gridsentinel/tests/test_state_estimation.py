import numpy as np

from gridsentinel.config import SimConfig
from gridsentinel.grid.simulator import simulate
from gridsentinel.grid.state_estimation import DCStateEstimator
from gridsentinel.attacks.fdi import stealth_fdi, naive_fdi


def _sim():
    return simulate(SimConfig(system="ieee14", timesteps=120, seed=3))


def test_clean_data_mostly_passes():
    sim = _sim()
    est = DCStateEstimator(sim.model, alpha=0.99)
    alarms = sum(est.estimate(sim.Z[t]).alarm for t in range(sim.T))
    assert alarms / sim.T < 0.08                     # ~1% expected, allow slack


def test_stealth_attack_preserves_residual():
    """The residue-preserving attack must NOT change the chi-square statistic."""
    sim = _sim()
    est = DCStateEstimator(sim.model, alpha=0.99)
    rng = np.random.default_rng(1)
    for t in range(0, sim.T, 20):
        clean = est.estimate(sim.Z[t])
        z_atk, c = stealth_fdi(sim.Z[t], model=sim.model, strength=0.3, rng=rng)
        atk = est.estimate(z_atk)
        assert abs(atk.j_statistic - clean.j_statistic) < 1e-6
        # but the estimated state is shifted by exactly c
        assert np.allclose(atk.x_hat - clean.x_hat, c, atol=1e-6)


def test_naive_attack_raises_residual():
    sim = _sim()
    est = DCStateEstimator(sim.model, alpha=0.99)
    rng = np.random.default_rng(2)
    worse = 0
    for t in range(0, sim.T, 10):
        clean = est.estimate(sim.Z[t]).j_statistic
        atk = est.estimate(naive_fdi(sim.Z[t], strength=0.3, rng=rng)).j_statistic
        worse += atk > clean
    assert worse >= 10                               # almost always increases J
