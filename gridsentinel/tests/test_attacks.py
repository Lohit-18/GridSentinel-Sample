import numpy as np

from gridsentinel.config import AttackConfig, SimConfig
from gridsentinel.grid.simulator import simulate
from gridsentinel.attacks.scenarios import generate_campaign, ATTACK_TYPES


def test_campaign_shapes_and_labels():
    sim = simulate(SimConfig(system="ieee14", timesteps=800, seed=5))
    camp = generate_campaign(sim, AttackConfig(seed=9))
    assert camp.Z_obs.shape == sim.Z.shape
    assert camp.y.shape[0] == camp.T
    assert set(np.unique(camp.y)).issubset({0, 1})
    assert 0.05 < camp.y.mean() < 0.4                 # roughly the requested fraction


def test_all_attack_types_representable():
    sim = simulate(SimConfig(system="ieee14", timesteps=3000, seed=6))
    camp = generate_campaign(sim, AttackConfig(attack_fraction=0.3, seed=1))
    present = set(camp.summary())
    # at least several distinct families show up in a large campaign
    assert len(present & set(ATTACK_TYPES)) >= 5


def test_attacks_change_measurements():
    sim = simulate(SimConfig(system="ieee14", timesteps=600, seed=7))
    camp = generate_campaign(sim, AttackConfig(seed=2))
    atk_idx = np.where(camp.y == 1)[0]
    # on average, attacked steps differ from the clean measurements
    diff = np.abs(camp.Z_obs[atk_idx] - sim.Z[atk_idx]).sum(axis=1)
    assert (diff > 1e-6).mean() > 0.6
