"""Cause probable d'une alerte : contributions SHAP regroupées par famille physique."""

import pandas as pd
import shap

from src.features import preparer_X

# Une feature appartient à la famille de sa mesure d'origine (ex. vibration_max_24h -> vibration)
FAMILLES_FEATURES: dict[str, list[str]] = {
    "Usure et entretien": ["heures_depuis_changement_outil", "heures_depuis_maintenance"],
    "Surchauffe": ["temperature_process", "ecart_thermique"],
    "Surcharge mécanique": ["couple", "vibration"],
    "Environnement / réseau": ["concentration_poussiere", "niveau_tension", "indicateur_coupure", "humidite", "precipitation_mm"],
}


def famille_de(feature: str) -> str | None:
    for famille, mesures in FAMILLES_FEATURES.items():
        if any(feature == m or feature.startswith(f"{m}_") for m in mesures):
            return famille
    return None


def _valeur(v):
    if pd.isna(v):
        return None
    return v if isinstance(v, str) else float(v)


class ExplicateurCause:
    def __init__(self, modele_calibre):
        # CalibratedClassifierCV(FrozenEstimator(lgbm)) : on explique le LightGBM sous-jacent
        self._explainer = shap.TreeExplainer(modele_calibre.estimator.estimator)

    def cause_probable(self, df: pd.DataFrame) -> list[dict]:
        """
        Par ligne : la famille dont les contributions positives (vers la panne) sont les plus fortes.
        Une feature qui pousse vers la normalité ne peut pas être la cause d'une alerte.
        """
        X = preparer_X(df)
        familles = [famille_de(col) for col in X.columns]
        resultats = []
        for ligne in self._explainer.shap_values(X):
            scores = dict.fromkeys(FAMILLES_FEATURES, 0.0)
            for famille, valeur in zip(familles, ligne):
                if famille:
                    scores[famille] += max(0.0, float(valeur))
            principale, score = max(scores.items(), key=lambda kv: kv[1])
            resultats.append(
                {
                    "cause_principale": principale if score > 0 else "Aucun signal dominant",
                    "score": round(score, 4),
                    "detail": {f: round(s, 4) for f, s in scores.items()},
                }
            )
        return resultats

    def contributions(self, ligne: pd.DataFrame, n: int = 4) -> list[dict]:
        """Les n features qui poussent le plus vers la panne sur une ligne : nom, famille, valeur, contribution."""
        X = preparer_X(ligne)
        valeurs = self._explainer.shap_values(X)[0]
        top = sorted(zip(X.columns, valeurs), key=lambda kv: kv[1], reverse=True)[:n]
        return [
            {
                "feature": f,
                "famille": famille_de(f),
                "valeur": _valeur(X.iloc[0][f]),
                "contribution": round(float(v), 3),
            }
            for f, v in top
            if v > 0
        ]
