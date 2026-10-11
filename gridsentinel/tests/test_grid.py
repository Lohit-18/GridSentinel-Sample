import numpy as np
import pytest

from gridsentinel.grid.ieee_systems import load_system, available_systems
from gridsentinel.grid.state_estimation import (
    build_measurement_model, dc_power_flow, DCStateEstimator,
)


@pytest.mark.parametrize("name", ["ieee9", "ieee14", "synth:20"])
def test_system_loads(name):
    sys = load_system(name)
    assert sys.n_bus >= 9 if name != "synth:20" else sys.n_bus == 20
    assert sys.n_branch >= sys.n_bus - 1            # at least a spanning tree


def test_susceptance_matrix_symmetric():
    sys = load_system("ieee14")
    B = sys.susceptance_matrix()
    assert np.allclose(B, B.T)
    assert np.allclose(B.sum(axis=1), 0, atol=1e-9)  # rows sum to zero


def test_dc_power_flow_balances():
    sys = load_system("ieee14")
    rng = np.random.default_rng(0)
    inj = rng.normal(0, 0.3, sys.n_bus)
    inj[sys.slack] -= inj.sum()                      # make it balanced
    theta = dc_power_flow(sys, inj)
    assert theta[sys.slack] == 0.0
    assert np.isfinite(theta).all()


def test_measurement_model_redundant():
    sys = load_system("ieee14")
    model = build_measurement_model(sys)
    assert model.m > model.n_state                   # redundancy required
    est = DCStateEstimator(model)
    assert est.dof == model.m - model.n_state
    assert est.threshold > 0


def test_available_systems():
    names = available_systems()
    assert "ieee14" in names and "synth:<n>" in names
