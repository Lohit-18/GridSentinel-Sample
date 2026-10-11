import numpy as np

from gridsentinel.config import AttackConfig, DetectConfig, SimConfig
from gridsentinel.grid.simulator import simulate
from gridsentinel.attacks.scenarios import generate_campaign
from gridsentinel.detect.pipeline import DetectionPipeline

_FAST = DetectConfig(rf_estimators=60, gb_estimators=50, iso_estimators=60, mlp_hidden=(32,))


def test_save_load_roundtrip(tmp_path):
    sim = simulate(SimConfig(system="ieee9", timesteps=900, seed=1))
    camp = generate_campaign(sim, AttackConfig(seed=11))
    pipe = DetectionPipeline(sim.model, _FAST).fit(camp)
    res1 = pipe.predict(camp)

    path = str(tmp_path / "pipe.joblib")
    pipe.save(path)
    loaded = DetectionPipeline.load(path)
    res2 = loaded.predict(camp)

    assert np.allclose(res1.ensemble_score, res2.ensemble_score)
    assert 0.0 < loaded.ensemble.operating_threshold < 1.0


def test_predict_requires_fit():
    sim = simulate(SimConfig(system="ieee9", timesteps=300, seed=1))
    camp = generate_campaign(sim, AttackConfig(seed=1))
    pipe = DetectionPipeline(sim.model, _FAST)
    try:
        pipe.predict(camp)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
