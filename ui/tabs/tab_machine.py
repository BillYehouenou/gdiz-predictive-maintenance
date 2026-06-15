import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ui.constants import FEAT_COLS, STEPS_MAP
from ui.helpers import badge, plot_axis, plot_layout, rgba
from ui.queries import get_predictor, q_machine_last, q_machine_list, q_machine_ts_full


def render(C: dict) -> None:
    machines = q_machine_list()
    m_labels = [f"Machine #{mid[-2:]}" for mid in machines]

    top_l, top_r = st.columns([3, 1])
    with top_l:
        sel_idx = st.selectbox("Sélectionner une machine", range(len(m_labels)), format_func=lambda i: m_labels[i])
    with top_r:
        st.markdown("<div style='margin-top:1.6rem;'>", unsafe_allow_html=True)
        sel_horizon = st.segmented_control(
            "Horizon",
            options=list(STEPS_MAP.keys()),
            default="24h",
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    if sel_horizon is None:
        sel_horizon = "24h"
    steps = STEPS_MAP[sel_horizon]
    sel_id = machines[sel_idx]

    snap = q_machine_last(sel_id)
    df_ts = q_machine_ts_full(sel_id, steps)

    # RF probability on latest snapshot
    model_prob: float | None = None
    try:
        predictor = get_predictor()
        feat_row = pd.DataFrame([{k: snap[k] for k in FEAT_COLS}])
        model_prob = float(predictor.predict(feat_row)["failure_probability"].iloc[0])
    except Exception:
        model_prob = None

    prob_val = model_prob if model_prob is not None else 0.0
    if prob_val < 0.02:
        p_color, p_label = C["green"], "Normal"
    elif prob_val < 0.15:
        p_color, p_label = C["orange"], "Surveillance"
    else:
        p_color, p_label = C["red"], "Risque élevé"

    st.markdown("<br>", unsafe_allow_html=True)

    # Santé de la machine
    wear = snap["tool_wear"]
    _calib_note = "Modèle non disponible" if model_prob is None else "RF Baseline · En calibration"

    col_risk, col_m1, col_m2, col_m3 = st.columns([1.25, 1, 1, 1])

    with col_risk:
        st.markdown(
            f"""<div style='
                text-align:center;padding:1.1rem .5rem 1rem;
                border-radius:10px;
                background:{p_color}14;
                border:1px solid {p_color}35;
                height:100%;box-sizing:border-box;'>
              <div style='font-size:.75rem;color:{C["faint"]};letter-spacing:.04em;'>PROBABILITÉ PANNE · ML</div>
              <div style='font-size:2.6rem;font-weight:700;color:{p_color};line-height:1.15;margin:.2rem 0;'>{prob_val * 100:.1f}%</div>
              <div style='font-size:.85rem;color:{p_color};'>{p_label}</div>
              <div style='font-size:.68rem;color:{C["faint"]};margin-top:.5rem;'>{_calib_note}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with col_m1:
        st.metric("Usure outil", f"{wear:.1f} %")
        st.metric("Type machine", snap["machine_type"])

    with col_m2:
        st.metric("Temp. process", f"{snap['process_temperature']:.1f} °C")
        st.metric("Pannes totales", str(snap["n_failures"]))

    with col_m3:
        st.metric("Vibration", f"{snap['vibration']:.2f} mm/s")
        st.metric("Stab. tension", f"{snap['voltage_stability']:.1f} %")

    st.markdown("<br>", unsafe_allow_html=True)