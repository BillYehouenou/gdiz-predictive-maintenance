import logging

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.features import FENETRE_24H, enrichir_serie_inference
from ui.constants import FEAT_COLS, STEPS_MAP
from ui.helpers import axe_graphique, disposition_graphique
from ui.queries import (
    obtenir_explicateur,
    obtenir_predicteur,
    q_dernier_releve_machine,
    q_liste_machines,
    q_priorites_parc,
    q_serie_complete_machine,
)

logger = logging.getLogger(__name__)

FENETRE_AFFICHAGE = "30j"


def _predire_serie(predicteur, df_serie_tampon: pd.DataFrame, display_steps: int) -> pd.DataFrame:
    """
    Enrichit temporellement puis prédit sur toute la série, puis ne garde que les dernières lignes.
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


def _index_machine_prioritaire(machines: list[str]) -> int:
    """
    Machine sélectionnée par défaut = la plus prioritaire de "à traiter aujourd'hui"
    (ui/priorites.py), pas systématiquement la première de la liste.
    """
    try:
        priorites = q_priorites_parc()
        if not priorites.empty:
            return machines.index(priorites.iloc[0]["id_machine"])
    except Exception as e:
        logger.warning(f"Machine prioritaire indisponible, repli sur la première : {e}")
    return 0


def afficher(C: dict) -> None:
    machines = q_liste_machines()
    m_labels = [f"Machine #{mid[-2:]}" for mid in machines]

    index_selectionne = st.selectbox(
        "Sélectionner une machine",
        range(len(m_labels)),
        format_func=lambda i: m_labels[i],
        index=_index_machine_prioritaire(machines),
    )

    steps = STEPS_MAP[FENETRE_AFFICHAGE]
    id_selectionne = machines[index_selectionne]

    snap = q_dernier_releve_machine(id_selectionne)
    df_serie = q_serie_complete_machine(id_selectionne, steps)

    # Métriques agrégées sur la fenêtre d'affichage
    has_ts = len(df_serie) > 0
    wear_max = float(df_serie["usure_outil"].max()) if has_ts else snap["usure_outil"]
    temp_max = float(df_serie["temperature_process"].max()) if has_ts else snap["temperature_process"]
    vibr_max = float(df_serie["vibration"].max()) if has_ts else snap["vibration"]
    vstab_min = float(df_serie["stabilite_tension"].min()) if has_ts else snap["stabilite_tension"]
    torque_max = float(df_serie["couple"].max()) if has_ts else snap["couple"]
    n_fail_period = int(df_serie["panne"].sum()) if has_ts else 0
    poussiere_max = float(df_serie["concentration_poussiere"].max()) if has_ts else snap["concentration_poussiere"]
    encrassement_max = float(df_serie["encrassement"].max()) if has_ts else snap["encrassement"]

    # Prédictions sur la série complète (enrichie temporellement)
    probabilite_pic: float = 0.0
    horodatage_pic = None
    probabilite_actuelle: float | None = None
    df_predictions = pd.DataFrame()
    cause_probable: str | None = None
    predicteur = None

    try:
        predicteur = obtenir_predicteur()
        # +FENETRE_24H lignes de tampon : sans elles, les features glissantes des premières lignes
        # de la fenêtre affichée seraient tronquées
        df_serie_tampon = q_serie_complete_machine(id_selectionne, steps + FENETRE_24H)
        df_predictions = _predire_serie(predicteur, df_serie_tampon, steps)
        if len(df_predictions) > 0:
            peak_idx = int(df_predictions["probabilite_panne"].idxmax())
            probabilite_pic = float(df_predictions.loc[peak_idx, "probabilite_panne"])
            horodatage_pic = df_predictions.loc[peak_idx, "timestamp"]
            probabilite_actuelle = float(df_predictions.iloc[-1]["probabilite_panne"])
            if probabilite_actuelle >= predicteur.threshold:
                enrichi = enrichir_serie_inference(df_serie_tampon)
                feat_df = enrichi[[k for k in FEAT_COLS if k in enrichi.columns]].dropna()
                if len(feat_df) > 0:
                    cause_probable = obtenir_explicateur().cause_probable(feat_df.iloc[[-1]])[0]["cause_principale"]
    except Exception as e:
        logger.warning(f"Prédiction indisponible pour {id_selectionne} : {e}")

    # On affiche la probailité actuelle. Un pic à 63 % survenu il y a 3
    # semaines mais retombé depuis ne doit pas afficher "risque élevé" aujourd'hui :
    # Seuils dérivés du seuil de décision réel du modèle : avec un
    # seuil optimal typiquement bas (pannes rares, ~3 % de base), des bornes fixes
    # afficheraient "Normal" pour une machine que le modèle alerte réellement.
    probabilite_affichee = probabilite_actuelle if probabilite_actuelle is not None else 0.0
    threshold = predicteur.threshold if predicteur is not None else 0.5
    if probabilite_affichee < threshold:
        p_color, p_label = C["green"], "Normal"
    elif probabilite_affichee < threshold * 3:
        p_color, p_label = C["orange"], "Surveillance"
    else:
        p_color, p_label = C["red"], "Risque élevé"

    _peak_line = (
        f"Pic sur 30j : {probabilite_pic * 100:.1f}% le {pd.Timestamp(horodatage_pic).strftime('%d/%m %H:%M')}"
        if horodatage_pic is not None
        else "Modèle non disponible"
    )
    _cause_html = (
        f'<div style="font-size:.72rem;color:{p_color};margin-top:.3rem;font-weight:600;">Cause probable : {cause_probable}</div>'
        if cause_probable
        else ""
    )

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
              <div style='font-size:.75rem;color:{C["faint"]};letter-spacing:.04em;'>PROBABILITE DE PANNE DANS 5 JOURS</div>
              <div style='font-size:2.6rem;font-weight:700;color:{p_color};
                line-height:1.15;margin:.2rem 0;'>{probabilite_affichee * 100:.1f}%</div>
              <div style='font-size:.85rem;color:{p_color};'>{p_label}</div>
              <div style='font-size:.68rem;color:{C["faint"]};margin-top:.3rem;'>{_peak_line}</div>{_cause_html}
            </div>""",
            unsafe_allow_html=True,
        )

    with col_m1:
        st.metric(f"Usure max - {FENETRE_AFFICHAGE}", f"{wear_max:.1f} %", help="Maximum d'usure outil observé sur la période")
        st.metric(
            f"Couple max - {FENETRE_AFFICHAGE}", f"{torque_max:.1f} Nm", help="Couple maximal — indicateur de surcharge mécanique"
        )
        st.metric(
            f"Poussière max - {FENETRE_AFFICHAGE}",
            f"{poussiere_max:.0f} µg/m³",
            help="Concentration de poussière maximale observée",
        )

    with col_m2:
        st.metric(
            f"Temp. process max - {FENETRE_AFFICHAGE}", f"{temp_max:.1f} °C", help="Température process maximale sur la période"
        )
        st.metric(f"Vibration max - {FENETRE_AFFICHAGE}", f"{vibr_max:.2f} mm/s", help="Vibration maximale enregistrée")
        st.metric(
            f"Encrassement max - {FENETRE_AFFICHAGE}", f"{encrassement_max:.0f} %", help="Encrassement maximal du process observé"
        )

    with col_m3:
        st.metric("Type machine", snap["type_machine"])
        st.metric(f"Pannes - {FENETRE_AFFICHAGE}", str(n_fail_period), help="Nombre de pannes constatées sur la période")
        st.metric(f"Stab. tension min - {FENETRE_AFFICHAGE}", f"{vstab_min:.1f} %", help="Stabilité réseau SBEE minimale")

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

        # Probabilité de panne, lissée avec spline
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
    else:
        st.caption("Courbe de risque indisponible - pas assez d'historique ou modèle inaccessible.")
