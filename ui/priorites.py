import pandas as pd

from src.features import calculer_features

COLONNES_PRIORITES = ["id_machine", "type_machine", "probabilite_panne", "cause_principale", "fenetre"]


def calculer_priorites_parc(df_historique: pd.DataFrame, predicteur, explicateur) -> pd.DataFrame:
    lignes = []
    for id_machine, g in df_historique.groupby("id_machine", sort=False):
        g = g.sort_values("timestamp")
        if len(g) < 2:
            continue
        derniere = calculer_features(g).iloc[[-1]]
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
