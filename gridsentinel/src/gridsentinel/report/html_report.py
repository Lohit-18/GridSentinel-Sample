"""Render a single self-contained HTML results report (figures embedded)."""
from __future__ import annotations

import base64
import datetime as _dt
import os
from typing import Any

from ..eval.experiments import ExperimentResult


def _b64(path: str) -> str:
    with open(path, "rb") as fh:
        return "data:image/png;base64," + base64.b64encode(fh.read()).decode()


def _fig(path: str, caption: str) -> str:
    if not path or not os.path.exists(path):
        return ""
    return (f'<figure><img alt="{caption}" src="{_b64(path)}"/>'
            f'<figcaption>{caption}</figcaption></figure>')


def _metrics_table(r: ExperimentResult) -> str:
    order = [d for d in r.to_json()["detector_order"] if d in r.detector_metrics]
    head = ("<tr><th>Detector</th><th>Precision</th><th>Recall</th><th>F1</th>"
            "<th>FPR</th><th>ROC-AUC</th><th>PR-AUC</th></tr>")
    rows = []
    for d in order:
        m = r.detector_metrics[d]
        cls = ' class="win"' if d == "gridsentinel_ensemble" else ""
        rows.append(
            f"<tr{cls}><td>{r.detector_labels[d]}</td>"
            f"<td>{m['precision']:.3f}</td><td>{m['recall']:.3f}</td>"
            f"<td><b>{m['f1']:.3f}</b></td><td>{m['false_positive_rate']:.3f}</td>"
            f"<td>{m.get('roc_auc', float('nan')):.3f}</td>"
            f"<td>{m.get('pr_auc', float('nan')):.3f}</td></tr>")
    return f"<table>{head}{''.join(rows)}</table>"


def _scaling_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    head = ("<tr><th>System</th><th>Buses</th><th>Meters</th>"
            "<th>Physics F1</th><th>Ensemble F1</th><th>Ensemble recall</th>"
            "<th>FPR</th><th>Train+eval (s)</th></tr>")
    body = "".join(
        f"<tr><td>{x['system']}</td><td>{x['buses']}</td><td>{x['measurements']}</td>"
        f"<td>{x['physics_f1']:.3f}</td><td><b>{x['ensemble_f1']:.3f}</b></td>"
        f"<td>{x['ensemble_recall']:.3f}</td><td>{x['ensemble_fpr']:.4f}</td>"
        f"<td>{x['fit_sec']}</td></tr>" for x in rows)
    return f"<table>{head}{body}</table>"


def _cards(r: ExperimentResult) -> str:
    h = r.headline()
    lat = r.latency["gridsentinel_ensemble"]
    items = [
        ("Ensemble F1", f"{h['ensemble_f1']:.2f}", f"baseline {h['physics_f1']:.2f}"),
        ("Attacks caught", f"{h['ensemble_recall']*100:.0f}%", f"baseline {h['physics_recall']*100:.0f}%"),
        ("False-alarm rate", f"{h['ensemble_fpr']*100:.1f}%", "of normal minutes"),
        ("ROC-AUC", f"{h['ensemble_roc_auc']:.3f}", "ranking quality"),
        ("Detection coverage", f"{lat['coverage']*100:.0f}%", "of attack windows"),
        ("Median latency", f"{lat['median_latency_steps']:.0f} steps",
         "from attack onset"),
    ]
    return "".join(
        f'<div class="card"><div class="k">{k}</div><div class="v">{v}</div>'
        f'<div class="s">{s}</div></div>' for k, v, s in items)


_CSS = """
:root{--bg:#f7f7f5;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#b8b7b2;
--line:#e6e6e3;--blue:#2a78d6;--accent:#1baf7a;--win:#eef6ef;--shadow:0 1px 3px rgba(0,0,0,.07)}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--surface:#1f1f1e;--ink:#f0efec;
--ink2:#c3c2b7;--muted:#6b6a64;--line:#33322f;--blue:#3987e5;--accent:#199e70;--win:#13251c;
--shadow:0 1px 3px rgba(0,0,0,.4)}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
line-height:1.55}.wrap{max-width:1020px;margin:0 auto;padding:40px 20px 80px}
header{border-bottom:3px solid var(--blue);padding-bottom:18px;margin-bottom:8px}
h1{font-size:2.1rem;margin:0 0 4px}.sub{color:var(--ink2);font-size:1.05rem;margin:0}
.badges{margin:14px 0 4px}.badge{display:inline-block;background:var(--surface);
border:1px solid var(--line);border-radius:999px;padding:3px 12px;font-size:.8rem;
color:var(--ink2);margin:0 6px 6px 0}.ethics{background:var(--win);border:1px solid var(--line);
border-radius:10px;padding:10px 14px;font-size:.88rem;color:var(--ink2);margin:16px 0}
h2{font-size:1.35rem;margin:38px 0 14px;padding-bottom:6px;border-bottom:1px solid var(--line)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:20px 0}
.card{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:16px;
box-shadow:var(--shadow)}.card .k{font-size:.78rem;color:var(--ink2);text-transform:uppercase;
letter-spacing:.04em}.card .v{font-size:1.9rem;font-weight:700;color:var(--blue);margin:2px 0}
.card .s{font-size:.8rem;color:var(--muted)}figure{margin:18px 0;background:var(--surface);
border:1px solid var(--line);border-radius:12px;padding:14px;box-shadow:var(--shadow)}
figure img{width:100%;height:auto;border-radius:6px;display:block}
figcaption{color:var(--ink2);font-size:.85rem;margin-top:10px;text-align:center}
table{width:100%;border-collapse:collapse;margin:14px 0;font-size:.9rem;background:var(--surface);
border:1px solid var(--line);border-radius:12px;overflow:hidden}
th,td{padding:9px 12px;text-align:center;border-bottom:1px solid var(--line)}
th{background:var(--bg);color:var(--ink2);font-weight:600;text-align:center}
td:first-child,th:first-child{text-align:left}tr.win td{background:var(--win);font-weight:600}
footer{margin-top:50px;color:var(--muted);font-size:.8rem;border-top:1px solid var(--line);
padding-top:16px}.lead{color:var(--ink2);max-width:72ch}
"""


def build_report(result: ExperimentResult, figures: dict[str, str],
                 scaling_rows: list[dict[str, Any]] | None,
                 out_path: str) -> str:
    """Write the HTML report and return its path."""
    scaling_rows = scaling_rows or []
    h = result.headline()
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    ds = result.dataset

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>GridSentinel — Results Report</title><style>{_CSS}</style></head>
<body><div class="wrap">
<header>
  <h1>GridSentinel &mdash; Results Report</h1>
  <p class="sub">Machine-learning detection of false data injection in a smart-grid substation
  ({result.system.upper()} test system)</p>
  <div class="badges">
    <span class="badge">Generated {now}</span>
    <span class="badge">{ds['measurements']} meters · {ds['state_dim']} states</span>
    <span class="badge">{ds['n_features']} features</span>
    <span class="badge">{ds['train_steps']} train / {ds['test_steps']} test steps</span>
    <span class="badge">test attack rate {ds['test_attack_rate']*100:.0f}%</span>
  </div>
  <div class="ethics">🔒 <b>Defensive &amp; fully synthetic.</b> Every grid, measurement and
  attack in this report is simulated in software. No real substation, device or network is
  involved. Attacks exist only to generate labelled data for training the detector.</div>
</header>

<p class="lead">GridSentinel learns the normal rhythm of a power grid and flags injected false
sensor data &mdash; including the stealthy, residue-preserving attack that the classical
chi-square bad-data detector is mathematically blind to. The ensemble lifts attack recall from
<b>{h['physics_recall']*100:.0f}%</b> (physics baseline) to <b>{h['ensemble_recall']*100:.0f}%</b>
while holding the false-alarm rate at <b>{h['ensemble_fpr']*100:.1f}%</b>.</p>

<div class="cards">{_cards(result)}</div>

<h2>The money shot: a stealth attack in action</h2>
<p class="lead">Top: the classical detector's statistic stays below its alarm line throughout the
attack &mdash; it never fires. Bottom: GridSentinel's ensemble probability jumps on the same
window, because the injected data forced an abnormal, instantaneous shift in the estimated grid
state.</p>
{_fig(figures.get('timeseries_detection',''), 'Stealth false-data injection: invisible to physics, obvious to the ensemble.')}

<h2>Every detector, side by side</h2>
{_metrics_table(result)}
{_fig(figures.get('detector_comparison',''), 'F1 and recall for each detector on the unseen test campaign.')}

<h2>Where the baseline is blind</h2>
{_fig(figures.get('per_attack_recall',''), 'Per-attack recall. The baseline misses stealth FDI and replay almost entirely.')}

<h2>Ranking quality</h2>
{_fig(figures.get('roc_pr',''), 'ROC and precision–recall curves.')}

<h2>Confusion matrices</h2>
{_fig(figures.get('confusion_matrices',''), 'Raw counts: baseline vs. ensemble on the test set.')}

<h2>What the model actually looks at</h2>
<p class="lead">Feature importance, coloured by domain. The protocol and state-dynamics features
are exactly the ones that expose the attacks the physics layer cannot see.</p>
{_fig(figures.get('feature_importance',''), 'Top features by random-forest importance.')}

{"<h2>Scaling across grid sizes</h2>" + _scaling_table(scaling_rows) if scaling_rows else ""}

<footer>
GridSentinel v1.0 · Apache-2.0 · reproducible with <code>python -m gridsentinel.cli demo</code>.
Results regenerate deterministically from the configured random seeds. This is a research and
education project; it is not a certified protection relay.
</footer>
</div></body></html>"""

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write(html)
    return out_path
