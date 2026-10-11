"""GridSentinel interactive dashboard (Streamlit).

Run locally:   streamlit run dashboard/app.py
Deploy free:   push to GitHub, then share.streamlit.io -> point at this file.
"""
from __future__ import annotations

import os
import sys

# make the package importable whether installed or run from the repo
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from gridsentinel.config import AttackConfig, DetectConfig, SimConfig
from gridsentinel.grid.simulator import simulate
from gridsentinel.attacks.scenarios import generate_campaign
from gridsentinel.detect.pipeline import DetectionPipeline
from gridsentinel.eval import metrics as M

BLUE, ORANGE, AQUA, RED, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#52514e"

st.set_page_config(page_title="GridSentinel", page_icon="⚡", layout="wide")


@st.cache_data(show_spinner=True)
def run(system: str, steps: int, attack_fraction: float, stealth: float,
        noise: float, seed: int):
    train_sim = simulate(SimConfig(system=system, timesteps=steps,
                                   meter_noise_std=noise, seed=seed))
    test_sim = simulate(SimConfig(system=system, timesteps=steps,
                                  meter_noise_std=noise, seed=seed + 1))
    tcfg = dict(attack_fraction=attack_fraction, stealth_strength=stealth)
    train = generate_campaign(train_sim, AttackConfig(seed=seed + 11, **tcfg))
    test = generate_campaign(test_sim, AttackConfig(seed=seed + 22, **tcfg))
    pipe = DetectionPipeline(train_sim.model, DetectConfig()).fit(train)
    res = pipe.predict(test)
    return {
        "minutes": test_sim.minutes, "res": res,
        "ens_m": M.binary_metrics(res.y_true, res.ensemble_pred, res.ensemble_score),
        "phy_m": M.binary_metrics(res.y_true, res.physics_pred, res.physics_score),
        "per_atk_e": M.per_attack_recall(res.atype, res.ensemble_pred),
        "per_atk_p": M.per_attack_recall(res.atype, res.physics_pred),
        "lat": M.detection_latency(res.y_true, res.ensemble_pred),
    }


st.title("⚡ GridSentinel")
st.caption("Machine-learning detection of false data injection in a smart-grid substation "
           "· defensive · fully synthetic · no real device involved")

with st.sidebar:
    st.header("Scenario")
    system = st.selectbox("Grid system", ["ieee14", "ieee9", "synth:30"], index=0)
    steps = st.slider("Timesteps (per split)", 600, 2600, 1400, 200)
    attack_fraction = st.slider("Attack fraction", 0.05, 0.4, 0.18, 0.01)
    stealth = st.slider("Stealth strength", 0.05, 0.6, 0.25, 0.05)
    noise = st.slider("Meter noise (σ)", 0.002, 0.02, 0.008, 0.001)
    seed = st.number_input("Seed", 0, 9999, 1)
    go_btn = st.button("Run detection", type="primary", use_container_width=True)

if go_btn or "cache_ok" not in st.session_state:
    st.session_state["cache_ok"] = True
    data = run(system, steps, attack_fraction, stealth, noise, int(seed))
else:
    data = run(system, steps, attack_fraction, stealth, noise, int(seed))

res = data["res"]
ens, phy = data["ens_m"], data["phy_m"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Ensemble F1", f"{ens['f1']:.2f}", f"baseline {phy['f1']:.2f}")
c2.metric("Attacks caught", f"{ens['recall']*100:.0f}%", f"baseline {phy['recall']*100:.0f}%")
c3.metric("False-alarm rate", f"{ens['false_positive_rate']*100:.1f}%")
c4.metric("ROC-AUC", f"{ens.get('roc_auc', float('nan')):.3f}")

# --- time series ---------------------------------------------------------- #
st.subheader("Detection timeline")
centre = int(np.where(res.atype == "stealth_fdi")[0].mean()) if (res.atype == "stealth_fdi").any() else int(res.y_true.argmax())
lo, hi = max(0, centre - 80), min(len(res.y_true), centre + 80)
t = np.arange(lo, hi)
fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=res.j_stat[lo:hi] / res.threshold, name="χ² J / threshold",
                         line=dict(color=ORANGE, width=2)))
fig.add_trace(go.Scatter(x=t, y=res.ensemble_score[lo:hi], name="ensemble P(attack)",
                         line=dict(color=BLUE, width=2)))
fig.add_hline(y=1.0, line_dash="dash", line_color=INK2)
for i in t:
    if res.y_true[i]:
        fig.add_vrect(x0=i - 0.5, x1=i + 0.5, fillcolor=RED, opacity=0.08, line_width=0)
fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                  legend=dict(orientation="h"), yaxis_title="score (shaded = true attack)")
st.plotly_chart(fig, use_container_width=True)

# --- per-attack recall ---------------------------------------------------- #
left, right = st.columns(2)
with left:
    st.subheader("Per-attack recall")
    atks = sorted(set(data["per_atk_e"]) | set(data["per_atk_p"]))
    bar = go.Figure()
    bar.add_trace(go.Bar(x=atks, y=[data["per_atk_p"].get(a, 0) for a in atks],
                         name="Physics χ²", marker_color=ORANGE))
    bar.add_trace(go.Bar(x=atks, y=[data["per_atk_e"].get(a, 0) for a in atks],
                         name="Ensemble", marker_color=BLUE))
    bar.update_layout(barmode="group", height=340, margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h"), yaxis_title="recall")
    st.plotly_chart(bar, use_container_width=True)
with right:
    st.subheader("Confusion (ensemble)")
    cm = [[ens["tn"], ens["fp"]], [ens["fn"], ens["tp"]]]
    hm = go.Figure(go.Heatmap(z=cm, x=["pred normal", "pred attack"],
                              y=["actual normal", "actual attack"],
                              colorscale="Blues", text=cm, texttemplate="%{text}",
                              showscale=False))
    hm.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(hm, use_container_width=True)
    st.caption(f"Coverage {data['lat']['coverage']*100:.0f}% of attack windows · "
               f"median latency {data['lat']['median_latency_steps']:.0f} steps")

st.divider()
st.caption("GridSentinel v1.0 · Apache-2.0 · This is a research/education project, not a "
           "certified protection relay.")
