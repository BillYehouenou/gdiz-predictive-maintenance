import pandas as pd

from src.features import enrichir_serie_inference
from ui.constants import FEAT_COLS

COLONNES_PRIORITES = ["id_machine", "type_machine", "probabilite_panne", "cause_principale", "fenetre"]


def calculer_priorites_parc(df_historique: pd.DataFrame, predicteur, explicateur) -> pd.DataFrame:
    """
    Vue "à traiter aujourd'hui" (point 34 roadmap) : pour chaque machine de
    `df_historique` (plusieurs lignes/machine, triées), prédit sur son dernier point
    disponible et ne garde que celles au-dessus du seuil de surveillance, triées par
    risque décroissant. La fenêtre affichée reste volontairement qualitative ("2 à
    5 jours", pas "panne dans 37h12") — le générateur physique lui-même rend le temps
    restant non prévisible avec précision une fois dans la zone à risque (rampe
    terminale de dégradation, cf. docs/data-generation.md) ; inventer une heure
    précise serait une fausse promesse, pas une feature.
    """
    lignes = []
    for id_machine, g in df_historique.groupby("id_machine", sort=False):
        g = g.sort_values("timestamp")
        if len(g) < 2:
            continue
        enrichi = enrichir_serie_inference(g)
        feat_df = enrichi[[k for k in FEAT_COLS if k in enrichi.columns]].dropna()
        if feat_df.empty:
            continue

        derniere = feat_df.iloc[[-1]]
        proba = float(predicteur.predire(derniere)["probabilite_panne"].iloc[0])
        if proba < predicteur.threshold:
            continue

        cause = explicateur.cause_probable(derniere)[0]
        risque_eleve = proba >= predicteur.threshold * 3
        lignes.append(
            {
                "id_machine": id_machine,
                "type_machine": g["type_machine"].iloc[-1],
                "probabilite_panne": proba,
                "cause_principale": cause["cause_principale"],
                "fenetre": "Risque élevé — à traiter en priorité (2 à 5 jours)"
                if risque_eleve
                else "À surveiller (dans la semaine)",
            }
        )

    if not lignes:
        return pd.DataFrame(columns=COLONNES_PRIORITES)
    return pd.DataFrame(lignes).sort_values("probabilite_panne", ascending=False).reset_index(drop=True)
