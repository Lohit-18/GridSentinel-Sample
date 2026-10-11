"""Generate the showcase figures from an :class:`ExperimentResult`.

Palette and mark choices follow a validated, colour-blind-safe scheme: one hue
carries one meaning, the baseline and the ensemble keep the same two colours in
every chart, grids and axes stay recessive, and bars are directly labelled.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..eval.experiments import ExperimentResult

PALETTE = {
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
    "magenta": "#e87ba4", "green": "#008300", "violet": "#6250d6", "red": "#e34948",
    "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e6e6e3",
    "surface": "#fcfcfb", "muted": "#b8b7b2",
}
_DOMAIN_COLOR = {
    "estimation": PALETTE["blue"], "state": PALETTE["orange"],
    "measurement": PALETTE["aqua"], "spatial": PALETTE["violet"],
    "protocol": PALETTE["magenta"], "other": PALETTE["muted"],
}


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": PALETTE["surface"],
        "axes.facecolor": PALETTE["surface"],
        "savefig.facecolor": PALETTE["surface"],
        "axes.edgecolor": PALETTE["ink2"], "axes.linewidth": 0.8,
        "axes.grid": True, "grid.color": PALETTE["grid"], "grid.linewidth": 0.8,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "text.color": PALETTE["ink"], "axes.labelcolor": PALETTE["ink2"],
        "xtick.color": PALETTE["ink2"], "ytick.color": PALETTE["ink2"],
        "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
        "font.family": "DejaVu Sans", "figure.dpi": 120,
    })


def _save(fig, outdir: str, name: str) -> str:
    path = os.path.join(outdir, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
def fig_detector_comparison(r: ExperimentResult, outdir: str) -> str:
    _style()
    order = [d for d in r.to_json()["detector_order"] if d in r.detector_metrics]
    labels = [r.detector_labels[d].replace(" (baseline)", "\n(baseline)") for d in order]
    f1 = [r.detector_metrics[d]["f1"] for d in order]
    rec = [r.detector_metrics[d]["recall"] for d in order]
    x = np.arange(len(order)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.8))
    b1 = ax.bar(x - w / 2, f1, w, label="F1 score", color=PALETTE["blue"], zorder=3)
    b2 = ax.bar(x + w / 2, rec, w, label="Recall", color=PALETTE["orange"], zorder=3)
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                    f"{bar.get_height():.2f}", ha="center", va="bottom",
                    fontsize=8.5, color=PALETTE["ink2"])
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.08); ax.set_ylabel("score")
    ax.set_title("Detection quality by detector (unseen test campaign)")
    ax.legend(frameon=False, ncol=2, loc="lower left")
    return _save(fig, outdir, "detector_comparison.png")


def fig_per_attack(r: ExperimentResult, outdir: str) -> str:
    _style()
    ph = r.per_attack["physics_chi2"]; en = r.per_attack["gridsentinel_ensemble"]
    attacks = sorted(set(ph) | set(en))
    nice = {a: a.replace("_", " ") for a in attacks}
    x = np.arange(len(attacks)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(x - w / 2, [ph.get(a, 0) for a in attacks], w,
           label="Physics χ² baseline", color=PALETTE["orange"], zorder=3)
    ax.bar(x + w / 2, [en.get(a, 0) for a in attacks], w,
           label="GridSentinel ensemble", color=PALETTE["blue"], zorder=3)
    for i, a in enumerate(attacks):
        ax.text(i - w / 2, ph.get(a, 0) + 0.02, f"{ph.get(a,0):.2f}", ha="center",
                fontsize=8, color=PALETTE["ink2"])
        ax.text(i + w / 2, en.get(a, 0) + 0.02, f"{en.get(a,0):.2f}", ha="center",
                fontsize=8, color=PALETTE["ink2"])
    ax.set_xticks(x); ax.set_xticklabels([nice[a] for a in attacks], rotation=20, ha="right")
    ax.set_ylim(0, 1.1); ax.set_ylabel("recall (detection rate)")
    ax.set_title("Per-attack recall — where the baseline is blind")
    ax.legend(frameon=False, loc="lower center", ncol=2)
    return _save(fig, outdir, "per_attack_recall.png")


def fig_curves(r: ExperimentResult, outdir: str) -> str:
    _style()
    fig, (axr, axp) = plt.subplots(1, 2, figsize=(10.5, 4.6))
    styl = [("gridsentinel_ensemble", PALETTE["blue"], "GridSentinel ensemble"),
            ("random_forest", PALETTE["aqua"], "Random Forest"),
            ("physics_chi2", PALETTE["orange"], "Physics χ²")]
    for key, col, lab in styl:
        fpr, tpr = r.curves["roc"][key]
        axr.plot(fpr, tpr, color=col, lw=2, label=lab)
        rec, prec = r.curves["pr"][key]
        axp.plot(rec, prec, color=col, lw=2, label=lab)
    axr.plot([0, 1], [0, 1], "--", color=PALETTE["muted"], lw=1)
    axr.set_xlabel("false positive rate"); axr.set_ylabel("true positive rate")
    axr.set_title("ROC"); axr.legend(frameon=False, fontsize=9, loc="lower right")
    axp.set_xlabel("recall"); axp.set_ylabel("precision")
    axp.set_title("Precision–Recall"); axp.legend(frameon=False, fontsize=9, loc="lower left")
    axp.set_ylim(0, 1.03)
    return _save(fig, outdir, "roc_pr.png")


def fig_timeseries(r: ExperimentResult, outdir: str) -> str:
    _style()
    s = r.sample_window
    t = np.arange(len(s["j_stat"]))
    y = np.array(s["y_true"])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    # shade attacked region(s)
    for ax in (a1, a2):
        in_atk = False
        for i, v in enumerate(y):
            if v and not in_atk:
                start = i; in_atk = True
            if in_atk and (not v or i == len(y) - 1):
                ax.axvspan(start, i, color=PALETTE["red"], alpha=0.10, zorder=0)
                in_atk = False
    a1.plot(t, s["j_stat"], color=PALETTE["orange"], lw=1.8, label="χ² statistic J")
    a1.axhline(s["j_threshold"], ls="--", color=PALETTE["ink2"], lw=1, label="χ² alarm threshold")
    a1.set_ylabel("J statistic"); a1.set_title("Physics χ² detector — stealth attack stays under the line")
    a1.legend(frameon=False, fontsize=9, loc="upper right")
    a2.plot(t, s["ensemble_score"], color=PALETTE["blue"], lw=1.8, label="ensemble attack probability")
    a2.axhline(s["ensemble_threshold"], ls="--", color=PALETTE["ink2"], lw=1, label="operating threshold")
    a2.fill_between(t, 0, 1, where=y.astype(bool), color=PALETTE["red"], alpha=0.0)
    a2.set_ylim(0, 1.03); a2.set_ylabel("P(attack)"); a2.set_xlabel("time step")
    a2.set_title("GridSentinel ensemble — fires on the same window")
    a2.legend(frameon=False, fontsize=9, loc="upper right")
    # one annotation
    a1.text(0.01, 0.92, "shaded = true attack", transform=a1.transAxes,
            fontsize=8.5, color=PALETTE["red"])
    return _save(fig, outdir, "timeseries_detection.png")


def fig_feature_importance(r: ExperimentResult, outdir: str, top: int = 13) -> str:
    _style()
    feats = r.feature_importances[:top][::-1]
    names = [f[0].replace("_", " ") for f in feats]
    vals = [f[1] for f in feats]
    doms = [f[2] for f in feats]
    colors = [_DOMAIN_COLOR.get(d, PALETTE["muted"]) for d in doms]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.barh(names, vals, color=colors, zorder=3)
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.01, i, f"{v:.3f}", va="center",
                fontsize=8, color=PALETTE["ink2"])
    ax.set_xlabel("random-forest importance")
    ax.set_title("Which signals catch the attacks (coloured by domain)")
    seen = []
    handles = []
    for d in ["estimation", "state", "measurement", "spatial", "protocol"]:
        if d in doms:
            handles.append(plt.Rectangle((0, 0), 1, 1, color=_DOMAIN_COLOR[d]))
            seen.append(d)
    ax.legend(handles, seen, frameon=False, fontsize=9, loc="lower right")
    ax.margins(x=0.12)
    return _save(fig, outdir, "feature_importance.png")


def fig_confusion(r: ExperimentResult, outdir: str) -> str:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, key, title in [(axes[0], "physics_chi2", "Physics χ² baseline"),
                           (axes[1], "gridsentinel_ensemble", "GridSentinel ensemble")]:
        mt = r.detector_metrics[key]
        cm = np.array([[mt["tn"], mt["fp"]], [mt["fn"], mt["tp"]]], dtype=float)
        norm = cm / cm.sum()
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(cm[i,j])}", ha="center", va="center",
                        color=PALETTE["ink"] if norm[i, j] < 0.5 else "white",
                        fontsize=12, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["normal", "attack"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["normal", "attack"])
        ax.set_xlabel("predicted"); ax.set_ylabel("actual")
        ax.set_title(f"{title}\nF1={mt['f1']:.2f}  recall={mt['recall']:.2f}", fontsize=11)
        ax.grid(False)
    return _save(fig, outdir, "confusion_matrices.png")


def generate_all_figures(r: ExperimentResult, outdir: str) -> dict[str, str]:
    """Render every figure into ``outdir`` and return name -> path."""
    os.makedirs(outdir, exist_ok=True)
    return {
        "detector_comparison": fig_detector_comparison(r, outdir),
        "per_attack_recall": fig_per_attack(r, outdir),
        "roc_pr": fig_curves(r, outdir),
        "timeseries_detection": fig_timeseries(r, outdir),
        "feature_importance": fig_feature_importance(r, outdir),
        "confusion_matrices": fig_confusion(r, outdir),
    }
