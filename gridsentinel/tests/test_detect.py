import numpy as np

from gridsentinel.config import AttackConfig, DetectConfig, SimConfig
from gridsentinel.grid.simulator import simulate
from gridsentinel.attacks.scenarios import generate_campaign
from gridsentinel.detect.pipeline import DetectionPipeline
from gridsentinel.eval.metrics import binary_metrics, per_attack_recall

_FAST = DetectConfig(rf_estimators=80, gb_estimators=60, iso_estimators=80,
                     mlp_hidden=(32,))


def _fit_predict():
    train_sim = simulate(SimConfig(system="ieee14", timesteps=1200, seed=1))
    test_sim = simulate(SimConfig(system="ieee14", timesteps=1000, seed=2))
    train = generate_campaign(train_sim, AttackConfig(seed=11))
    test = generate_campaign(test_sim, AttackConfig(seed=22))
    pipe = DetectionPipeline(train_sim.model, _FAST).fit(train)
    return pipe, pipe.predict(test), test


def test_ensemble_beats_physics_overall():
    _, res, _ = _fit_predict()
    ens = binary_metrics(res.y_true, res.ensemble_pred)
    phy = binary_metrics(res.y_true, res.physics_pred)
    assert ens["f1"] > phy["f1"] + 0.1
    assert ens["recall"] > phy["recall"]
    assert ens["false_positive_rate"] < 0.05


def test_ensemble_catches_stealth_that_physics_misses():
    _, res, _ = _fit_predict()
    pe = per_attack_recall(res.atype, res.ensemble_pred)
    pp = per_attack_recall(res.atype, res.physics_pred)
    # physics is essentially blind to the residue-preserving attack
    assert pp.get("stealth_fdi", 0) < 0.3
    # the ensemble catches most of them
    assert pe.get("stealth_fdi", 0) > 0.7


def test_scores_are_probabilities():
    _, res, _ = _fit_predict()
    assert res.ensemble_score.min() >= 0.0 and res.ensemble_score.max() <= 1.0
    assert res.ensemble_pred.shape == res.y_true.shape
