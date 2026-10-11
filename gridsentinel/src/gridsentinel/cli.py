"""GridSentinel command-line interface (``python -m gridsentinel.cli ...``)."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .config import AttackConfig, DetectConfig, SimConfig, set_global_seed
from .grid.simulator import simulate
from .grid.ieee_systems import available_systems
from .attacks.scenarios import generate_campaign
from .protocol.goose import measurements_to_goose, write_pcap
from .detect.pipeline import DetectionPipeline
from .eval.experiments import run_experiment, run_scaling_study

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    _RICH = True
    _c = Console()
except Exception:                                    # pragma: no cover
    _RICH = False
    _c = None


def _print(msg: str) -> None:
    (_c.print(msg) if _RICH else print(msg))


def _banner() -> None:
    art = ("[bold cyan]GridSentinel[/]  ·  smart-grid false-data-injection detector\n"
           "[dim]defensive · fully synthetic · free to run[/]")
    _print(Panel(art, box=box.ROUNDED, expand=False) if _RICH else
           "=== GridSentinel — smart-grid FDI detector (synthetic) ===")


def _metrics_table(result) -> None:
    if not _RICH:
        for d, m in result.detector_metrics.items():
            print(f"  {d:24s} F1={m['f1']:.3f} recall={m['recall']:.3f} "
                  f"FPR={m['false_positive_rate']:.3f}")
        return
    t = Table(title="Detector comparison (unseen test campaign)", box=box.SIMPLE_HEAVY)
    for col in ("Detector", "Precision", "Recall", "F1", "FPR", "ROC-AUC"):
        t.add_column(col, justify="right" if col != "Detector" else "left")
    for d in result.to_json()["detector_order"]:
        if d not in result.detector_metrics:
            continue
        m = result.detector_metrics[d]
        style = "bold green" if d == "gridsentinel_ensemble" else None
        t.add_row(result.detector_labels[d], f"{m['precision']:.3f}",
                  f"{m['recall']:.3f}", f"{m['f1']:.3f}",
                  f"{m['false_positive_rate']:.3f}",
                  f"{m.get('roc_auc', float('nan')):.3f}", style=style)
    _c.print(t)


# --------------------------------------------------------------------------- #
def cmd_demo(args) -> None:
    from .viz.plots import generate_all_figures
    from .report.html_report import build_report
    set_global_seed(args.seed)
    _banner()
    _print(f"[dim]Running full demo on {args.system} …[/]" if _RICH
           else f"Running full demo on {args.system} …")
    result = run_experiment(args.system, train_steps=args.train_steps,
                            test_steps=args.test_steps, verbose=True)
    os.makedirs(args.out, exist_ok=True)
    figs = generate_all_figures(result, os.path.join(args.out, "figures"))
    scaling = run_scaling_study() if args.scaling else []
    report = build_report(result, figs, scaling, os.path.join(args.out, "report.html"))
    with open(os.path.join(args.out, "results.json"), "w") as fh:
        json.dump(result.to_json(), fh, indent=2)
    _metrics_table(result)
    h = result.headline()
    _print(f"\n[bold green]✓[/] baseline recall {h['physics_recall']*100:.0f}% "
           f"→ ensemble recall {h['ensemble_recall']*100:.0f}% "
           f"(FPR {h['ensemble_fpr']*100:.1f}%)" if _RICH else
           f"OK baseline recall {h['physics_recall']*100:.0f}% -> "
           f"ensemble {h['ensemble_recall']*100:.0f}%")
    _print(f"  report  → {report}")
    _print(f"  figures → {os.path.join(args.out, 'figures')}/")
    _print(f"  json    → {os.path.join(args.out, 'results.json')}")


def cmd_evaluate(args) -> None:
    set_global_seed(args.seed)
    _banner()
    result = run_experiment(args.system, train_steps=args.train_steps,
                            test_steps=args.test_steps, verbose=True)
    _metrics_table(result)


def cmd_scaling(args) -> None:
    set_global_seed(args.seed)
    _banner()
    rows = run_scaling_study(steps=args.steps)
    if _RICH:
        t = Table(title="Scaling study", box=box.SIMPLE_HEAVY)
        for col in ("System", "Buses", "Meters", "Physics F1", "Ensemble F1", "FPR", "Seconds"):
            t.add_column(col, justify="right" if col != "System" else "left")
        for x in rows:
            t.add_row(x["system"], str(x["buses"]), str(x["measurements"]),
                      f"{x['physics_f1']:.3f}", f"{x['ensemble_f1']:.3f}",
                      f"{x['ensemble_fpr']:.4f}", str(x["fit_sec"]))
        _c.print(t)


def cmd_simulate(args) -> None:
    set_global_seed(args.seed)
    _banner()
    sim = simulate(SimConfig(system=args.system, timesteps=args.steps))
    camp = generate_campaign(sim, AttackConfig())
    os.makedirs(args.out, exist_ok=True)
    # CSV of measurements + labels
    csv_path = os.path.join(args.out, "campaign.csv")
    header = ",".join([f"z{i}" for i in range(sim.model.m)]) + ",label,attack_type"
    with open(csv_path, "w") as fh:
        fh.write(header + "\n")
        for t in range(camp.T):
            row = ",".join(f"{v:.5f}" for v in camp.Z_obs[t])
            fh.write(f"{row},{camp.y[t]},{camp.atype[t]}\n")
    # pcap of GOOSE traffic
    msgs = [measurements_to_goose(camp.Z_obs[t], t_ms=int(sim.minutes[t] * 60000),
                                  st_num=int(camp.st_num[t]), sq_num=int(camp.sq_num[t]))
            for t in range(min(camp.T, 500))]
    pcap_path = os.path.join(args.out, "goose_traffic.pcap")
    nbytes = write_pcap(pcap_path, msgs)
    _print(f"  campaign  → {csv_path}  ({camp.T} steps, mix={camp.summary()})")
    _print(f"  goose pcap→ {pcap_path}  ({len(msgs)} frames, {nbytes} bytes)")


def cmd_info(args) -> None:
    _banner()
    _print("Available systems: " + ", ".join(available_systems()))
    _print("Attack families: naive_fdi, stealth_fdi, ramp_fdi, replay, goose_spoof, goose_flood")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gridsentinel", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("demo", help="run the full experiment + report + figures")
    d.add_argument("--system", default="ieee14")
    d.add_argument("--train-steps", type=int, default=2600)
    d.add_argument("--test-steps", type=int, default=2000)
    d.add_argument("--out", default="results")
    d.add_argument("--scaling", action="store_true", help="also run the scaling study")
    d.add_argument("--seed", type=int, default=1337)
    d.set_defaults(func=cmd_demo)

    e = sub.add_parser("evaluate", help="train + evaluate, print metrics")
    e.add_argument("--system", default="ieee14")
    e.add_argument("--train-steps", type=int, default=2600)
    e.add_argument("--test-steps", type=int, default=2000)
    e.add_argument("--seed", type=int, default=1337)
    e.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("scaling", help="sweep across grid sizes")
    s.add_argument("--steps", type=int, default=1200)
    s.add_argument("--seed", type=int, default=1337)
    s.set_defaults(func=cmd_scaling)

    sm = sub.add_parser("simulate", help="export a labelled campaign (CSV) + GOOSE pcap")
    sm.add_argument("--system", default="ieee14")
    sm.add_argument("--steps", type=int, default=1500)
    sm.add_argument("--out", default="results")
    sm.add_argument("--seed", type=int, default=1337)
    sm.set_defaults(func=cmd_simulate)

    i = sub.add_parser("info", help="list systems and attack families")
    i.set_defaults(func=cmd_info)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
