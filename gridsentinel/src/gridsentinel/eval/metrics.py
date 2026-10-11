"""Detection metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                   score: np.ndarray | None = None) -> dict[str, float]:
    """Precision / recall / F1 / accuracy / FPR / MCC (+ AUCs if scores given)."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    fpr = fp / (fp + tn) if fp + tn else 0.0
    out = {
        "precision": prec, "recall": rec, "f1": f1, "accuracy": acc,
        "false_positive_rate": fpr, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(set(y_pred)) > 1 else 0.0,
    }
    if score is not None and len(set(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, score))
        out["pr_auc"] = float(average_precision_score(y_true, score))
    return out


def per_attack_recall(atype: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Recall for each attack family (fraction of that family that was flagged)."""
    out: dict[str, float] = {}
    for at in sorted(set(atype)):
        if at == "normal":
            continue
        m = atype == at
        if m.sum():
            out[str(at)] = float(np.asarray(y_pred)[m].mean())
    return out


def detection_latency(y_true: np.ndarray, y_pred: np.ndarray
                      ) -> dict[str, float]:
    """Mean steps from each attack-window start to first detection.

    Windows never detected count toward ``coverage`` but not ``mean_latency``.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    latencies: list[int] = []
    detected = total = 0
    t = 0
    T = len(y_true)
    while t < T:
        if y_true[t] == 1:
            start = t
            while t < T and y_true[t] == 1:
                t += 1
            total += 1
            window = range(start, t)
            hit = next((i for i in window if y_pred[i] == 1), None)
            if hit is not None:
                detected += 1
                latencies.append(hit - start)
        else:
            t += 1
    return {
        "windows": total,
        "detected_windows": detected,
        "coverage": detected / total if total else 0.0,
        "mean_latency_steps": float(np.mean(latencies)) if latencies else float("nan"),
        "median_latency_steps": float(np.median(latencies)) if latencies else float("nan"),
    }


def roc_curve_points(y_true: np.ndarray, score: np.ndarray, n: int = 60):
    fpr, tpr, _ = roc_curve(y_true, score)
    idx = np.linspace(0, len(fpr) - 1, min(n, len(fpr))).astype(int)
    return fpr[idx].tolist(), tpr[idx].tolist()


def pr_curve_points(y_true: np.ndarray, score: np.ndarray, n: int = 60):
    prec, rec, _ = precision_recall_curve(y_true, score)
    idx = np.linspace(0, len(prec) - 1, min(n, len(prec))).astype(int)
    return rec[idx].tolist(), prec[idx].tolist()
