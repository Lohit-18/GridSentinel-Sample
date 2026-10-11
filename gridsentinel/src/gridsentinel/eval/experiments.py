"""Run the headline experiment and the scaling study, and package results."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import AttackConfig, DetectConfig, SimConfig
from ..attacks.scenarios import generate_campaign
from ..grid.simulator import simulate
from ..detect.pipeline import DetectionPipeline
from ..features.extractor import FEATURE_DOMAIN
from . import metrics as M

_DETECTOR_ORDER = [
    "physics_chi2", "isolation_forest", "mlp",
    "gradient_boost", "random_forest", "gridsentinel_ensemble",
]
_DETECTOR_LABELS = {
    "physics_chi2": "Physics χ² (baseline)",
    "isolation_forest": "Isolation Forest",
    "mlp": "Neural Net (MLP)",
    "gradient_boost": "Gradient Boosting",
    "random_forest": "Random Forest",
    "gridsentinel_ensemble": "GridSentinel Ensemble",
}


def _best_threshold(score: np.ndarray, y: np.ndarray) -> float:
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 37):
        pred = (score >= t).astype(int)
        mt = M.binary_metrics(y, pred)
        if mt["f1"] > best_f1:
            best_f1, best_t = mt["f1"], float(t)
    return best_t


@dataclass
class ExperimentResult:
    """Everything the report and plots need from one experiment."""

    system: str
    sim_config: dict[str, Any]
    attack_config: dict[str, Any]
    detect_config: dict[str, Any]
    dataset: dict[str, Any]
    detector_metrics: dict[str, dict[str, float]]
    detector_labels: dict[str, str]
    per_attack: dict[str, dict[str, float]]
    latency: dict[str, dict[str, float]]
    curves: dict[str, Any]
    feature_importances: list[tuple[str, float, str]]
    sample_window: dict[str, Any]
    timing_sec: dict[str, float]

    def headline(self) -> dict[str, float]:
        ph = self.detector_metrics["physics_chi2"]
        en = self.detector_metrics["gridsentinel_ensemble"]
        return {
            "physics_recall": ph["recall"], "physics_f1": ph["f1"],
            "ensemble_recall": en["recall"], "ensemble_f1": en["f1"],
            "ensemble_fpr": en["false_positive_rate"],
            "ensemble_roc_auc": en.get("roc_auc", float("nan")),
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "sim_config": self.sim_config,
            "attack_config": self.attack_config,
            "detect_config": self.detect_config,
            "dataset": self.dataset,
            "detector_order": _DETECTOR_ORDER,
            "detector_labels": self.detector_labels,
            "detector_metrics": self.detector_metrics,
            "per_attack": self.per_attack,
            "latency": self.latency,
            "curves": self.curves,
            "feature_importances": [
                {"name": n, "importance": imp, "domain": dom}
                for n, imp, dom in self.feature_importances
            ],
            "sample_window": self.sample_window,
            "timing_sec": self.timing_sec,
            "headline": self.headline(),
        }


def run_experiment(system: str = "ieee14",
                   train_steps: int = 2600,
                   test_steps: int = 2000,
                   sim_cfg: SimConfig | None = None,
                   atk_cfg: AttackConfig | None = None,
                   det_cfg: DetectConfig | None = None,
                   verbose: bool = True) -> ExperimentResult:
    """Train on one campaign, evaluate every detector on an unseen campaign."""
    det_cfg = det_cfg or DetectConfig()
    t0 = time.time()

    base_sim = sim_cfg or SimConfig()
    train_sim = simulate(SimConfig(system=system, timesteps=train_steps,
                                   meter_noise_std=base_sim.meter_noise_std, seed=1))
    test_sim = simulate(SimConfig(system=system, timesteps=test_steps,
                                  meter_noise_std=base_sim.meter_noise_std, seed=2))
    base_atk = atk_cfg or AttackConfig()
    train = generate_campaign(train_sim, AttackConfig(
        attack_fraction=base_atk.attack_fraction, mix=dict(base_atk.mix),
        stealth_strength=base_atk.stealth_strength,
        naive_strength=base_atk.naive_strength, seed=11))
    test = generate_campaign(test_sim, AttackConfig(
        attack_fraction=base_atk.attack_fraction, mix=dict(base_atk.mix),
        stealth_strength=base_atk.stealth_strength,
        naive_strength=base_atk.naive_strength, seed=22))
    t_data = time.time() - t0

    pipe = DetectionPipeline(train_sim.model, det_cfg, alpha=det_cfg.chi2_alpha).fit(train)
    t_fit = time.time() - t0 - t_data

    # --- collect scores on train (for thresholds) and test ----------------- #
    train_fm = pipe.features(train)
    train_bases = pipe.ensemble.base_scores(train_fm.X)
    train_ens = pipe.ensemble.score(train_fm.X, train_fm.j_stat / train_fm.threshold)

    res = pipe.predict(test)
    y = res.y_true
    score_map = {
        "isolation_forest": (train_bases["isolation_forest"], res.base_scores["isolation_forest"]),
        "mlp": (train_bases["mlp"], res.base_scores["mlp"]),
        "gradient_boost": (train_bases["gradient_boost"], res.base_scores["gradient_boost"]),
        "random_forest": (train_bases["random_forest"], res.base_scores["random_forest"]),
        "gridsentinel_ensemble": (train_ens, res.ensemble_score),
    }

    detector_metrics: dict[str, dict[str, float]] = {}
    pred_map: dict[str, np.ndarray] = {}

    # physics: statistical threshold, not tuned
    detector_metrics["physics_chi2"] = M.binary_metrics(y, res.physics_pred, res.physics_score)
    pred_map["physics_chi2"] = res.physics_pred

    for name, (tr_s, te_s) in score_map.items():
        if name == "gridsentinel_ensemble":
            thr = pipe.ensemble.operating_threshold
        else:
            thr = _best_threshold(tr_s, train_fm.y)
        pred = (te_s >= thr).astype(int)
        pred_map[name] = pred
        detector_metrics[name] = M.binary_metrics(y, pred, te_s)
        detector_metrics[name]["threshold"] = float(thr)

    per_attack = {
        "physics_chi2": M.per_attack_recall(res.atype, res.physics_pred),
        "gridsentinel_ensemble": M.per_attack_recall(res.atype, res.ensemble_pred),
        "random_forest": M.per_attack_recall(res.atype, pred_map["random_forest"]),
    }
    latency = {
        "physics_chi2": M.detection_latency(y, res.physics_pred),
        "gridsentinel_ensemble": M.detection_latency(y, res.ensemble_pred),
    }

    curves = {
        "roc": {
            "gridsentinel_ensemble": M.roc_curve_points(y, res.ensemble_score),
            "random_forest": M.roc_curve_points(y, res.base_scores["random_forest"]),
            "physics_chi2": M.roc_curve_points(y, res.physics_score),
        },
        "pr": {
            "gridsentinel_ensemble": M.pr_curve_points(y, res.ensemble_score),
            "random_forest": M.pr_curve_points(y, res.base_scores["random_forest"]),
            "physics_chi2": M.pr_curve_points(y, res.physics_score),
        },
    }

    imp = pipe.ensemble.feature_importances_
    names = train_fm.names
    order = np.argsort(imp)[::-1]
    feat_imp = [(names[i], float(imp[i]), FEATURE_DOMAIN.get(names[i], "other"))
                for i in order]

    # pick a window containing a stealth attack for the time-series plot
    stealth_idx = np.where(res.atype == "stealth_fdi")[0]
    centre = int(stealth_idx[len(stealth_idx) // 2]) if stealth_idx.size else res.y_true.argmax()
    lo, hi = max(0, centre - 60), min(len(y), centre + 60)
    sample_window = {
        "minutes": test_sim.minutes[lo:hi].tolist(),
        "j_stat": res.j_stat[lo:hi].tolist(),
        "j_threshold": float(res.threshold),
        "ensemble_score": res.ensemble_score[lo:hi].tolist(),
        "ensemble_threshold": float(pipe.ensemble.operating_threshold),
        "y_true": res.y_true[lo:hi].tolist(),
        "atype": res.atype[lo:hi].tolist(),
    }

    result = ExperimentResult(
        system=system,
        sim_config=train_sim.config.to_dict(),
        attack_config={"attack_fraction": base_atk.attack_fraction,
                       "mix": base_atk.normalised_mix(),
                       "stealth_strength": base_atk.stealth_strength,
                       "naive_strength": base_atk.naive_strength},
        detect_config=det_cfg.__dict__ | {"mlp_hidden": list(det_cfg.mlp_hidden)},
        dataset={
            "train_steps": train.T, "test_steps": test.T,
            "n_features": train_fm.X.shape[1],
            "measurements": train_sim.model.m,
            "state_dim": train_sim.model.n_state,
            "train_mix": train.summary(), "test_mix": test.summary(),
            "test_attack_rate": float(y.mean()),
        },
        detector_metrics=detector_metrics,
        detector_labels=_DETECTOR_LABELS,
        per_attack=per_attack,
        latency=latency,
        curves=curves,
        feature_importances=feat_imp,
        sample_window=sample_window,
        timing_sec={"data": t_data, "fit": t_fit, "total": time.time() - t0},
    )
    if verbose:
        h = result.headline()
        print(f"[{system}] physics F1={h['physics_f1']:.2f} "
              f"-> ensemble F1={h['ensemble_f1']:.2f} "
              f"(recall {h['physics_recall']:.2f} -> {h['ensemble_recall']:.2f}, "
              f"FPR {h['ensemble_fpr']:.3f})  {result.timing_sec['total']:.1f}s")
    return result


def run_scaling_study(systems: list[str] | None = None,
                      steps: int = 1200) -> list[dict[str, Any]]:
    """Quick F1/timing sweep across grid sizes for the scaling table."""
    systems = systems or ["ieee9", "ieee14", "synth:30", "synth:57"]
    rows = []
    for sysname in systems:
        r = run_experiment(sysname, train_steps=steps, test_steps=steps,
                           verbose=False)
        h = r.headline()
        rows.append({
            "system": sysname,
            "buses": r.dataset["state_dim"] + 1,
            "measurements": r.dataset["measurements"],
            "physics_f1": round(h["physics_f1"], 3),
            "ensemble_f1": round(h["ensemble_f1"], 3),
            "ensemble_recall": round(h["ensemble_recall"], 3),
            "ensemble_fpr": round(h["ensemble_fpr"], 4),
            "fit_sec": round(r.timing_sec["total"], 1),
        })
        print(f"  scaling {sysname:9s} ensemble F1={rows[-1]['ensemble_f1']:.3f} "
              f"({rows[-1]['fit_sec']}s)")
    return rows
