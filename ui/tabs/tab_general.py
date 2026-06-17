import plotly.graph_objects as go
import streamlit as st

from ui.constants import PERIOD_MAP, PERIOD_PREV_LABEL, PERIOD_PREV_MAP
from ui.helpers import cost_delta_html, fmt_fcfa, plot_axis, plot_layout, rgba
from ui.queries import (
    q_cost_period,
    q_cost_timeline,
    q_failures_by_type_cause_period,
    q_healthy_machines_pct,
    q_kpis_period,
)


def render(C: dict) -> None:
    sel_period = st.segmented_control(
        "Période",
        options=list(PERIOD_MAP.keys()),
        default="T4 2025",
        label_visibility="collapsed",
    )
    if sel_period is None:
        sel_period = "T4 2025"
    date_from, date_to = PERIOD_MAP[sel_period]

    kpi = q_kpis_period(date_from, date_to)
    cost_curr = q_cost_period(date_from, date_to)
    healthy_pct = q_healthy_machines_pct(date_from, date_to)

    prev_dates = PERIOD_PREV_MAP[sel_period]
    prev_label = PERIOD_PREV_LABEL[sel_period]
    cost_prev_total = q_cost_period(*prev_dates)["total_cost"] if prev_dates else None

    # 5 KPI clés
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "Machines saines",
        f"{healthy_pct:.1f} %",
        help="Part des machines sans aucune panne sur la période",
    )
    c2.metric("MTBF moyen", f"{kpi['mtbf_h']:,.0f} h", help="Temps moyen entre deux défaillances")
    c3.metric(
        "Fenêtres PM",
        f"{kpi['pm_windows']:,}",
        help="Fenêtres 24h avec usure 82–90% sans défaillance.",
    )
    c4.metric("Délestages (30j)", f"{kpi['blackouts_30d']:,}", help="Événements SBEE sur les 30 derniers jours du dataset")

    _help_cost = (
        f"C_panne = C_production + C_technique + C_qualité · {cost_curr['n_events']} événements. "
        f"Prod: {fmt_fcfa(cost_curr['c_production'])} · "
        f"Tech: {fmt_fcfa(cost_curr['c_technique'])} · "
        f"Qualité: {fmt_fcfa(cost_curr['c_qualite'])}"
    )
    c5.metric("Coût pannes subies", fmt_fcfa(cost_curr["total_cost"]), help=_help_cost)
    if cost_prev_total is not None:
        delta_cost = cost_curr["total_cost"] - cost_prev_total
        c5.markdown(cost_delta_html(C, delta_cost, prev_label), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        f"<p style='color:{C['muted']};font-size:1.3rem;font-weight:660;letter-spacing:0;margin:0;'>Pannes par type de machine · {sel_period}</p>",
        unsafe_allow_html=True,
    )
    df_fail = q_failures_by_type_cause_period(date_from, date_to)
    types = ["L", "M", "H"]
    cause_colors = {
        "Mécanique": rgba(C["orange"], 0.72),
        "Électrique": rgba(C["blue"], 0.72),
        "Thermique": rgba(C["teal"], 0.72),
    }
    fig_bar = go.Figure()
    for cause, color in cause_colors.items():
        sub = df_fail[df_fail["cause"] == cause].set_index("machine_type")
        y_vals = [float(sub.loc[t, "n"]) if t in sub.index else 0 for t in types]
        fig_bar.add_trace(go.Bar(name=cause, x=types, y=y_vals, marker_color=color, marker_line_width=0))

    fig_bar.update_layout(
        **plot_layout(
            C,
            barmode="stack",
            height=300,
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.18,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(0,0,0,0)",
                font=dict(size=14, color=C["muted"]),
            ),
        ),
        xaxis=dict(**plot_axis(C), title=""),
        yaxis=dict(**plot_axis(C), title="Pannes"),
    )
    st.plotly_chart(fig_bar, width='stretch', config={"displayModeBar": False})

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        f"<p style='color:{C['muted']};font-size:1.3rem;font-weight:660;letter-spacing:0;margin:0;'>Coût des pannes · {sel_period}</p>",
        unsafe_allow_html=True,
    )
    df_timeline = q_cost_timeline(date_from, date_to)

    if df_timeline.empty:
        st.caption("Aucune panne sur la période.")
    else:
        fig_cost = go.Figure()
        fig_cost.add_trace(go.Scatter(
            x=df_timeline["period"],
            y=df_timeline["cost"],
            name="Coût",
            mode="lines+markers",
            line=dict(color=C["red"], width=2, shape="spline"),
            marker=dict(size=6, color=C["red"]),
            fill="tozeroy",
            fillcolor=rgba(C["red"], 0.08),
            hovertemplate="%{x|%b %Y}<br>%{y:,.0f} FCFA<extra></extra>",
        ))
        fig_cost.update_layout(
            **plot_layout(C, height=260, plot_bgcolor="rgba(0,0,0,0)", showlegend=False),
            xaxis=dict(**plot_axis(C)),
            yaxis=dict(**plot_axis(C), title="Coût"),
        )
        st.plotly_chart(fig_cost, width='stretch', config={"displayModeBar": False})
