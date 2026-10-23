"""
Baselines : le modèle doit battre une règle triviale et une régression logistique sur le recall événementiel.
Usage : python scripts/run_baseline_comparison.py
"""

import logging

from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CIBLE
from src.donnees import charger, decouper
from src.evaluation import grille_seuils, metriques_evenementielles, trouver_seuil_pour_budget_fa
from src.features import FEATURES, preparer_X
from src.train import BUDGET_FAUSSES_ALERTES_PAR_MACHINE_JOUR, PAS_HORIZON

logger = logging.getLogger(__name__)


def evaluer(nom: str, probas_validation, probas_test, validation, test) -> None:
    grille = grille_seuils(probas_validation)
    seuil, _ = trouver_seuil_pour_budget_fa(
        validation, probas_validation, grille, PAS_HORIZON, BUDGET_FAUSSES_ALERTES_PAR_MACHINE_JOUR
    )
    m = metriques_evenementielles(test, probas_test, seuil, PAS_HORIZON)
    logger.info(
        f"{nom} : recall test {m['recall_evenementiel']:.3f} ({m['nb_attrapees']}/{m['nb_pannes']}), "
        f"FA/machine/jour {m['taux_fausses_alertes_par_machine_jour']:.3f}"
    )


def main():
    train, validation, test = decouper(charger())

    # Règle triviale : heures d'outil au-dessus d'un seuil
    colonne = "heures_depuis_changement_outil"
    evaluer("Règle triviale", validation[colonne].to_numpy(), test[colonne].to_numpy(), validation, test)

    # Régression logistique : imputation, normalisation, one-hot du type de machine
    numeriques = [f for f in FEATURES if f != "type_machine"]
    rl = make_pipeline(
        make_column_transformer(
            (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeriques),
            (OneHotEncoder(handle_unknown="ignore"), ["type_machine"]),
        ),
        LogisticRegression(max_iter=3000, class_weight="balanced"),
    ).fit(preparer_X(train), train[CIBLE])
    probas_val, probas_test = (rl.predict_proba(preparer_X(d))[:, 1] for d in (validation, test))
    evaluer("Régression logistique", probas_val, probas_test, validation, test)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
