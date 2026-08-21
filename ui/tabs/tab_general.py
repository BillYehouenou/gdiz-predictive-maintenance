import plotly.graph_objects as go
import streamlit as st

from ui.constants import PERIOD_MAP, PERIOD_PREV_LABEL, PERIOD_PREV_MAP
from ui.helpers import axe_graphique, disposition_graphique, formater_fcfa, html_delta_cout, rgba
from ui.queries import (
    q_chronologie_cout,
    q_cout_periode,
    q_kpis_periode,
    q_pannes_par_type_cause_periode,
    q_pourcentage_machines_saines,
)


def afficher(C: dict) -> None:
    sel_period = st.segmented_control(
        "Période",
        options=list(PERIOD_MAP.keys()),
        default="T4 2025",
        label_visibility="collapsed",
    )
    if sel_period is None:
        sel_period = "T4 2025"
    date_debut, date_fin = PERIOD_MAP[sel_period]

    kpi = q_kpis_periode(date_debut, date_fin)
    cost_curr = q_cout_periode(date_debut, date_fin)
    healthy_pct = q_pourcentage_machines_saines(date_debut, date_fin)

    prev_dates = PERIOD_PREV_MAP[sel_period]
    prev_label = PERIOD_PREV_LABEL[sel_period]
    cost_prev_total = q_cout_periode(*prev_dates)["total_cost"] if prev_dates else None

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
    c4.metric("Délestages (30j)", f"{kpi['blackouts_30d']:,}", help="Événements SBEE sur les 30 derniers jours du jeu_donnees")

    _help_cost = (
        f"C_panne = C_production + C_technique + C_qualité · {cost_curr['n_events']} événements. "
        f"Prod: {formater_fcfa(cost_curr['c_production'])} · "
        f"Tech: {formater_fcfa(cost_curr['c_technique'])} · "
        f"Qualité: {formater_fcfa(cost_curr['c_qualite'])}"
    )
    c5.metric("Coût pannes subies", formater_fcfa(cost_curr["total_cost"]), help=_help_cost)
    if cost_prev_total is not None:
        delta_cost = cost_curr["total_cost"] - cost_prev_total
        c5.markdown(html_delta_cout(C, delta_cost, prev_label), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        f"<p style='color:{C['muted']};font-size:1.3rem;font-weight:660;"
        f"letter-spacing:0;margin:0;'>Pannes par type de machine · {sel_period}</p>",
        unsafe_allow_html=True,
    )
    df_fail = q_pannes_par_type_cause_periode(date_debut, date_fin)
    types = ["L", "M", "H"]
    # Causes façon AI4I 2020 : TWF=usure, HDF=dissipation thermique, OSF=surcharge
    # mécanique, PWF=électrique, RNF=aléatoire (sans précurseur, par construction).
    # Libellés en français + sigle technique entre parenthèses (cohérent avec tab_about).
    cause_labels = {
        "TWF": "Usure (TWF)",
        "HDF": "Surchauffe (HDF)",
        "OSF": "Surcharge méca. (OSF)",
        "PWF": "Électrique (PWF)",
        "RNF": "Aléatoire (RNF)",
    }
    cause_colors = {
        "TWF": rgba(C["orange"], 0.72),
        "HDF": rgba(C["teal"], 0.72),
        "OSF": rgba(C["red"], 0.72),
        "PWF": rgba(C["blue"], 0.72),
        "RNF": rgba(C["green"], 0.72),
    }
    fig_bar = go.Figure()
    for cause, color in cause_colors.items():
        sub = df_fail[df_fail["cause"] == cause].set_index("type_machine")
        y_vals = [float(sub.loc[t, "n"]) if t in sub.index else 0 for t in types]
        fig_bar.add_trace(go.Bar(name=cause_labels[cause], x=types, y=y_vals, marker_color=color, marker_line_width=0))

    fig_bar.update_layout(
        **disposition_graphique(
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
        xaxis=dict(**axe_graphique(C), title=""),
        yaxis=dict(**axe_graphique(C), title="Pannes"),
    )
    st.plotly_chart(fig_bar, width="stretch", config={"displayModeBar": False})

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        f"<p style='color:{C['muted']};font-size:1.3rem;font-weight:660;"
        f"letter-spacing:0;margin:0;'>Coût des pannes · {sel_period}</p>",
        unsafe_allow_html=True,
    )
    df_timeline = q_chronologie_cout(date_debut, date_fin)

    if df_timeline.empty:
        st.caption("Aucune panne sur la période.")
    else:
        fig_cost = go.Figure()
        fig_cost.add_trace(
            go.Scatter(
                x=df_timeline["period"],
                y=df_timeline["cost"],
                name="Coût",
                mode="lines+markers",
                line=dict(color=C["red"], width=2, shape="spline"),
                marker=dict(size=6, color=C["red"]),
                fill="tozeroy",
                fillcolor=rgba(C["red"], 0.08),
                hovertemplate="%{x|%b %Y}<br>%{y:,.0f} FCFA<extra></extra>",
            )
        )
        fig_cost.update_layout(
            **disposition_graphique(C, height=260, plot_bgcolor="rgba(0,0,0,0)", showlegend=False),
            xaxis=dict(**axe_graphique(C)),
            yaxis=dict(**axe_graphique(C), title="Coût"),
        )
        st.plotly_chart(fig_cost, width="stretch", config={"displayModeBar": False})
