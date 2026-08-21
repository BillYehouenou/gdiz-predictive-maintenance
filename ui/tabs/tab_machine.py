import logging

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.features import FENETRE_120H, enrichir_serie_inference
from ui.constants import FEAT_COLS, STEPS_MAP
from ui.helpers import axe_graphique, disposition_graphique
from ui.queries import obtenir_predicteur, q_dernier_releve_machine, q_liste_machines, q_serie_complete_machine

logger = logging.getLogger(__name__)


def _predire_serie(predicteur, df_serie_tampon: pd.DataFrame, display_steps: int) -> pd.DataFrame:
    """
    Enrichit temporellement puis prédit sur toute la série, puis ne garde que les
    `display_steps` dernières lignes. `df_serie_tampon` doit contenir display_steps +
    FENETRE_120H lignes : sans ce tampon, les features glissantes (jusqu'à 120h) des
    premières lignes de la fenêtre affichée seraient calculées avec un historique
    tronqué (ex. usure_outil_delta_24h=0 par défaut faute de point 24h en arrière
    disponible), ce qui fausse la probabilité affichée selon la fenêtre choisie (24h/7j/30j).
    """
    df_enriched = enrichir_serie_inference(df_serie_tampon)
    feat_df = df_enriched[[k for k in FEAT_COLS if k in df_enriched.columns]].dropna()
    if len(feat_df) == 0:
        return pd.DataFrame()
    preds = predicteur.predire(feat_df)
    out = df_enriched.loc[feat_df.index, ["timestamp", "usure_outil", "vibration", "panne"]].copy()
    out["probabilite_panne"] = preds["probabilite_panne"].values
    out["prediction"] = preds["prediction"].values
    return out.reset_index(drop=True).tail(display_steps).reset_index(drop=True)


def afficher(C: dict) -> None:
    machines = q_liste_machines()
    m_labels = [f"Machine #{mid[-2:]}" for mid in machines]

    top_l, top_r = st.columns([3, 1])
    with top_l:
        index_selectionne = st.selectbox("Sélectionner une machine", range(len(m_labels)), format_func=lambda i: m_labels[i])
    with top_r:
        st.markdown("<div style='margin-top:1.6rem;'>", unsafe_allow_html=True)
        horizon_selectionne = st.segmented_control(
            "Horizon",
            options=list(STEPS_MAP.keys()),
            default="30j",
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    if horizon_selectionne is None:
        horizon_selectionne = "30j"
    steps = STEPS_MAP[horizon_selectionne]
    id_selectionne = machines[index_selectionne]

    snap = q_dernier_releve_machine(id_selectionne)
    df_serie = q_serie_complete_machine(id_selectionne, steps)

    # Métriques agrégées sur la fenêtre sélectionnée
    has_ts = len(df_serie) > 0
    wear_max = float(df_serie["usure_outil"].max()) if has_ts else snap["usure_outil"]
    temp_max = float(df_serie["temperature_process"].max()) if has_ts else snap["temperature_process"]
    vibr_max = float(df_serie["vibration"].max()) if has_ts else snap["vibration"]
    vstab_min = float(df_serie["stabilite_tension"].min()) if has_ts else snap["stabilite_tension"]
    torque_max = float(df_serie["couple"].max()) if has_ts else snap["couple"]
    n_fail_period = int(df_serie["panne"].sum()) if has_ts else 0

    # Prédictions sur la série complète (enrichie temporellement)
    probabilite_pic: float = 0.0
    horodatage_pic = None
    probabilite_actuelle: float | None = None
    df_predictions = pd.DataFrame()
    predicteur = None

    try:
        predicteur = obtenir_predicteur()
        # +FENETRE_120H lignes de tampon : sans elles, les features glissantes (jusqu'à
        # 120h) des premières lignes de la fenêtre affichée seraient tronquées (cf. _predire_serie).
        df_serie_tampon = q_serie_complete_machine(id_selectionne, steps + FENETRE_120H)
        df_predictions = _predire_serie(predicteur, df_serie_tampon, steps)
        if len(df_predictions) > 0:
            peak_idx = int(df_predictions["probabilite_panne"].idxmax())
            probabilite_pic = float(df_predictions.loc[peak_idx, "probabilite_panne"])
            horodatage_pic = df_predictions.loc[peak_idx, "timestamp"]
            probabilite_actuelle = float(df_predictions.iloc[-1]["probabilite_panne"])
    except Exception as e:
        logger.warning(f"Prédiction indisponible pour {id_selectionne} : {e}")

    # Risque pic = métrique principale pour la maintenance
    # Seuils dérivés du seuil de décision réel du modèle (probabilités calibrées,
    # cf. docs/model-decisions.md) plutôt que des bornes fixes 0,35/0,50 : avec un
    # seuil optimal typiquement bas (pannes rares, ~3 % de base), des bornes fixes
    # afficheraient "Normal" pour une machine que le modèle alerte réellement.
    probabilite_affichee = probabilite_pic
    threshold = predicteur.threshold if predicteur is not None else 0.5
    if probabilite_affichee < threshold:
        p_color, p_label = C["green"], "Normal"
    elif probabilite_affichee < threshold * 3:
        p_color, p_label = C["orange"], "Surveillance"
    else:
        p_color, p_label = C["red"], "Risque élevé"

    _peak_date = f"pic le {pd.Timestamp(horodatage_pic).strftime('%d/%m %H:%M')}" if horodatage_pic is not None else ""
    _calib_note = "Modèle non disponible" if probabilite_actuelle is None else "LightGBM · Probabilités calibrées (isotonique)"
    _h = horizon_selectionne

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
              <div style='font-size:.75rem;color:{C["faint"]};letter-spacing:.04em;'>RISQUE DE PANNE · ≤ 5 JOURS</div>
              <div style='font-size:2.6rem;font-weight:700;color:{p_color};
                line-height:1.15;margin:.2rem 0;'>{probabilite_affichee * 100:.1f}%</div>
              <div style='font-size:.85rem;color:{p_color};'>{p_label}</div>
              <div style='font-size:.68rem;color:{C["faint"]};margin-top:.3rem;'>pic sur les {_h} · {_peak_date}</div>
              <div style='font-size:.68rem;color:{C["faint"]};margin-top:.2rem;'>{_calib_note}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with col_m1:
        st.metric(f"Usure max · {_h}", f"{wear_max:.1f} %", help="Maximum d'usure outil observé sur la période")
        st.metric("Type machine", snap["type_machine"])
        st.metric(f"Couple max · {_h}", f"{torque_max:.1f} Nm", help="Couple maximal — indicateur de surcharge mécanique")

    with col_m2:
        st.metric(f"Temp. process max · {_h}", f"{temp_max:.1f} °C", help="Température process maximale sur la période")
        st.metric(f"Pannes · {_h}", str(n_fail_period), help="Nombre de pannes constatées sur la période sélectionnée")

    with col_m3:
        st.metric(f"Vibration max · {_h}", f"{vibr_max:.2f} mm/s", help="Vibration maximale enregistrée")
        st.metric(f"Stab. tension min · {_h}", f"{vstab_min:.1f} %", help="Stabilité réseau SBEE minimale")

    st.markdown("<br>", unsafe_allow_html=True)

    # Graphique : courbe de risque sur la fenêtre
    if len(df_predictions) > 0:
        st.markdown(
            f"<p style='color:{C['muted']};font-size:1.3rem;font-weight:660;letter-spacing:0;margin:0;'>Courbe de risque</p>",
            unsafe_allow_html=True,
        )

        prob_max_pct = max(float(df_predictions["probabilite_panne"].max()) * 100, 1.0)
        y_max = min(prob_max_pct * 1.40, 106)

        fig = go.Figure()

        # Probabilité de panne — lissée avec spline
        fig.add_trace(
            go.Scatter(
                x=df_predictions["timestamp"],
                y=df_predictions["probabilite_panne"] * 100,
                name="Risque",
                line=dict(color=C["red"], width=1.8, shape="spline"),
            )
        )

        # Marqueurs de pannes constatées
        failures = df_predictions[df_predictions["panne"] == 1]
        if len(failures) > 0:
            fig.add_trace(
                go.Scatter(
                    x=failures["timestamp"],
                    y=failures["probabilite_panne"] * 100,
                    mode="markers",
                    name="Panne constatée",
                    marker=dict(color=C["red"], size=9, symbol="x-thin", line=dict(width=2, color=C["red"])),
                )
            )

        fig.update_layout(
            **disposition_graphique(C, height=300, plot_bgcolor="rgba(0,0,0,0)", showlegend=False),
            xaxis=dict(**axe_graphique(C)),
            yaxis=dict(**axe_graphique(C), title="Risque", range=[0, y_max]),
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
