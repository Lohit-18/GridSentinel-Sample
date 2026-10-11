"""Minimal end-to-end example: simulate -> attack -> detect.

Run with:  python examples/quickstart.py
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from gridsentinel.config import AttackConfig, DetectConfig, SimConfig, set_global_seed
from gridsentinel.grid.simulator import simulate
from gridsentinel.attacks.scenarios import generate_campaign
from gridsentinel.detect.pipeline import DetectionPipeline
from gridsentinel.eval.metrics import binary_metrics, per_attack_recall


def main() -> None:
    set_global_seed(7)

    # 1) simulate a clean grid, then two labelled attack campaigns
    train_sim = simulate(SimConfig(system="ieee14", timesteps=1800, seed=1))
    test_sim = simulate(SimConfig(system="ieee14", timesteps=1500, seed=2))
    train = generate_campaign(train_sim, AttackConfig(seed=11))
    test = generate_campaign(test_sim, AttackConfig(seed=22))
    print("test campaign mix:", test.summary())

    # 2) train the detection pipeline
    pipe = DetectionPipeline(train_sim.model, DetectConfig()).fit(train)

    # 3) score the unseen campaign
    res = pipe.predict(test)
    ens = binary_metrics(res.y_true, res.ensemble_pred, res.ensemble_score)
    phy = binary_metrics(res.y_true, res.physics_pred, res.physics_score)

    print(f"\nPhysics  χ² baseline : recall={phy['recall']:.2f}  F1={phy['f1']:.2f}")
    print(f"GridSentinel ensemble: recall={ens['recall']:.2f}  F1={ens['f1']:.2f}  "
          f"FPR={ens['false_positive_rate']:.3f}")

    print("\nPer-attack recall (baseline vs ensemble):")
    pe = per_attack_recall(res.atype, res.ensemble_pred)
    pp = per_attack_recall(res.atype, res.physics_pred)
    for a in sorted(pe):
        print(f"  {a:12s}  {pp.get(a,0):.2f}  ->  {pe.get(a,0):.2f}")


if __name__ == "__main__":
    main()
