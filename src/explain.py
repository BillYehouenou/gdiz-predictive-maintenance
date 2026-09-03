import pandas as pd
import shap

# Quel phénomène physique pousse cette alerte ? (usure, surchauffe, surcharge...)
FAMILLES_FEATURES: dict[str, list[str]] = {
    "Usure de l'outil": ["usure_outil", "usure_outil_delta_24h", "usure_outil_delta_4h"],
    "Surchauffe": [
        "temperature_process",
        "temperature_process_max_24h",
        "temperature_process_moyenne_24h",
        "temperature_process_delta_4h",
        "ecart_thermique",
        "ecart_thermique_max_24h",
        "encrassement",
        "encrassement_max_24h",
    ],
    "Surcharge mécanique": [
        "couple",
        "couple_max_24h",
        "vibration",
        "vibration_max_24h",
        "vibration_moyenne_24h",
        "vibration_ecart_type_24h",
        "vibration_delta_4h",
    ],
    "Environnement / réseau": [
        "concentration_poussiere",
        "poussiere_moyenne_24h",
        "poussiere_max_24h",
        "niveau_tension",
        "stabilite_tension",
        "indicateur_coupure",
        "humidite",
        "indicateur_pluie",
    ],
}


class ExplicateurCause:
    """
    Explique une alerte par contribution SHAP agrégée en familles physiques
    SHAP répond à cette question : "qu'est-ce qui, dans les capteurs de 
    CETTE machine, pousse le modèle vers CETTE panne ?".
    """

    def __init__(self, pipeline_calibre):
        base = pipeline_calibre.calibrated_classifiers_[0].estimator
        self.preprocessor = base.named_steps["preprocessor"]
        self.lgbm = base.named_steps["classifier"]
        self._explainer = shap.TreeExplainer(self.lgbm)

    def cause_probable(self, X: pd.DataFrame) -> list[dict]:
        """
        Pour chaque ligne de X, la famille physique dont les features contribuent le
        plus positivement (vers panne) à la prédiction SHAP, et le détail par famille.
        Seules les contributions positives comptent : une feature qui pousse VERS la
        normalité ne peut pas être la "cause" d'une alerte.
        """
        X_transforme = self.preprocessor.transform(X)
        valeurs_shap = self._explainer.shap_values(X_transforme)
        colonnes = [c.split("__", 1)[-1] for c in X_transforme.columns]

        resultats = []
        for ligne in valeurs_shap:
            contributions = dict(zip(colonnes, ligne))
            scores_famille = {
                famille: sum(max(0.0, contributions.get(col, 0.0)) for col in colonnes_famille)
                for famille, colonnes_famille in FAMILLES_FEATURES.items()
            }
            famille_principale, score = max(scores_famille.items(), key=lambda kv: kv[1])
            resultats.append(
                {
                    "cause_principale": famille_principale if score > 0 else "Aucun signal dominant",
                    "score": round(float(score), 4),
                    "detail": {f: round(float(s), 4) for f, s in scores_famille.items()},
                }
            )
        return resultats
