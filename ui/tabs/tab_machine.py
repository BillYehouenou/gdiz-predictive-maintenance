import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.features import enrich_inference_series
from ui.constants import FEAT_COLS, STEPS_MAP
from ui.helpers import plot_axis, plot_layout
from ui.queries import get_predictor, q_machine_last, q_machine_list, q_machine_ts_full


def _predict_series(predictor, df_ts: pd.DataFrame) -> pd.DataFrame:
    """Enrichit temporellement puis prédit sur toute la série — retourne df avec failure_probability."""
    df_enriched = enrich_inference_series(df_ts)
    feat_df = df_enriched[[k for k in FEAT_COLS if k in df_enriched.columns]].dropna()
    if len(feat_df) == 0:
        return pd.DataFrame()
    preds = predictor.predict(feat_df)
    out = df_enriched.loc[feat_df.index, ["timestamp", "tool_wear", "vibration", "target"]].copy()
    out["failure_probability"] = preds["failure_probability"].values
    out["prediction"] = preds["prediction"].values
    return out.reset_index(drop=True)


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
            default="30j",
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    if sel_horizon is None:
        sel_horizon = "30j"
    steps = STEPS_MAP[sel_horizon]
    sel_id = machines[sel_idx]

    snap = q_machine_last(sel_id)
    df_ts = q_machine_ts_full(sel_id, steps)

    # Métriques agrégées sur la fenêtre sélectionnée
    has_ts = len(df_ts) > 0
    wear_max    = float(df_ts["tool_wear"].max())          if has_ts else snap["tool_wear"]
    temp_max    = float(df_ts["process_temperature"].max()) if has_ts else snap["process_temperature"]
    vibr_max    = float(df_ts["vibration"].max())          if has_ts else snap["vibration"]
    vstab_min   = float(df_ts["voltage_stability"].min())  if has_ts else snap["voltage_stability"]
    torque_max  = float(df_ts["torque"].max())             if has_ts else snap["torque"]
    n_fail_period = int(df_ts["target"].sum())             if has_ts else 0

    # Prédictions sur la série complète (enrichie temporellement)
    peak_prob: float = 0.0
    peak_ts = None
    current_prob: float | None = None
    df_pred = pd.DataFrame()

    try:
        predictor = get_predictor()
        df_pred = _predict_series(predictor, df_ts)
        if len(df_pred) > 0:
            peak_idx = int(df_pred["failure_probability"].idxmax())
            peak_prob = float(df_pred.loc[peak_idx, "failure_probability"])
            peak_ts = df_pred.loc[peak_idx, "timestamp"]
            current_prob = float(df_pred.iloc[-1]["failure_probability"])
    except Exception:
        pass

    # Risque pic = métrique principale pour la maintenance
    disp_prob = peak_prob
    if disp_prob < 0.35:
        p_color, p_label = C["green"], "Normal"
    elif disp_prob < 0.50:
        p_color, p_label = C["orange"], "Surveillance"
    else:
        p_color, p_label = C["red"], "Risque élevé"

    _peak_date = f"pic le {pd.Timestamp(peak_ts).strftime('%d/%m %H:%M')}" if peak_ts is not None else ""
    _calib_note = "Modèle non disponible" if current_prob is None else "LightGBM · Seuil F2 optimisé"
    _h = sel_horizon

    st.markdown("<br>", unsafe_allow_html=True)

    col_risk, col_m1, col_m2, col_m3 = st.columns([1.25, 1, 1, 1])

    with col_risk:
        st.markdown(
            f"""<div style='
                text-align:center;padding:1.1rem .5rem 1rem;
                border-radius:10px;
                background:{p_color}14;
                border:1px solid {p_color}35;
                height:100%;box-sizing:border-box;'>
              <div style='font-size:.75rem;color:{C["faint"]};letter-spacing:.04em;'>RISQUE MAX · {_h.upper()}</div>
              <div style='font-size:2.6rem;font-weight:700;color:{p_color};line-height:1.15;margin:.2rem 0;'>{disp_prob * 100:.1f}%</div>
              <div style='font-size:.85rem;color:{p_color};'>{p_label}</div>
              <div style='font-size:.68rem;color:{C["faint"]};margin-top:.3rem;'>{_peak_date}</div>
              <div style='font-size:.68rem;color:{C["faint"]};margin-top:.2rem;'>{_calib_note}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with col_m1:
        st.metric(f"Usure max · {_h}", f"{wear_max:.1f} %", help="Maximum d'usure outil observé sur la période")
        st.metric("Type machine", snap["machine_type"])
        st.metric(f"Couple max · {_h}", f"{torque_max:.1f} Nm", help="Couple maximal — indicateur de surcharge mécanique")

    with col_m2:
        st.metric(f"Temp. process max · {_h}", f"{temp_max:.1f} °C", help="Température process maximale sur la période")
        st.metric(f"Pannes · {_h}", str(n_fail_period), help="Nombre de pannes constatées sur la période sélectionnée")

    with col_m3:
        st.metric(f"Vibration max · {_h}", f"{vibr_max:.2f} mm/s", help="Vibration maximale enregistrée")
        st.metric(f"Stab. tension min · {_h}", f"{vstab_min:.1f} %", help="Stabilité réseau SBEE minimale")

    st.markdown("<br>", unsafe_allow_html=True)

    # Graphique : courbe de risque sur la fenêtre
    if len(df_pred) > 0:
        st.markdown(
            f"<p style='color:{C['muted']};font-size:1.3rem;font-weight:660;letter-spacing:0;margin:0;'>Courbe de risque</p>",
            unsafe_allow_html=True,
        )

        prob_max_pct = max(float(df_pred["failure_probability"].max()) * 100, 1.0)
        y_max = min(prob_max_pct * 1.40, 106)

        fig = go.Figure()

        # Probabilité de panne — lissée avec spline
        fig.add_trace(go.Scatter(
            x=df_pred["timestamp"],
            y=df_pred["failure_probability"] * 100,
            name="Risque",
            line=dict(color=C["red"], width=1.8, shape="spline"),
        ))

        # Marqueurs de pannes constatées
        failures = df_pred[df_pred["target"] == 1]
        if len(failures) > 0:
            fig.add_trace(go.Scatter(
                x=failures["timestamp"],
                y=failures["failure_probability"] * 100,
                mode="markers",
                name="Panne constatée",
                marker=dict(color=C["red"], size=9, symbol="x-thin", line=dict(width=2, color=C["red"])),
            ))

        fig.update_layout(
            **plot_layout(C, height=300, plot_bgcolor="rgba(0,0,0,0)", showlegend=False),
            xaxis=dict(**plot_axis(C)),
            yaxis=dict(**plot_axis(C), title="Risque", range=[0, y_max]),
        )
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
