import joblib
import pytest

from src.configloader import charger_config
from src.dataloader import ChargeurDonnees
from src.explain import FAMILLES_FEATURES, ExplicateurCause


@pytest.fixture(scope="module")
def explicateur():
    pipeline = joblib.load("models/model_pipeline.pkl")
    return ExplicateurCause(pipeline)


@pytest.fixture(scope="module")
def echantillon():
    config = charger_config()
    colonnes = config["jeu_donnees"]["features"]
    df = ChargeurDonnees().charger_donnees_brutes()
    return df[colonnes].sample(5, random_state=1)


def test_cause_probable_retourne_une_ligne_par_entree(explicateur, echantillon):
    resultats = explicateur.cause_probable(echantillon)
    assert len(resultats) == len(echantillon)


def test_cause_probable_famille_valide(explicateur, echantillon):
    resultats = explicateur.cause_probable(echantillon)
    familles_valides = set(FAMILLES_FEATURES) | {"Aucun signal dominant"}
    for r in resultats:
        assert r["cause_principale"] in familles_valides
        assert set(r["detail"]) == set(FAMILLES_FEATURES)


def test_cause_probable_detail_toujours_positif_ou_nul(explicateur, echantillon):
    """Seules les contributions qui poussent VERS la panne comptent — jamais négatif."""
    resultats = explicateur.cause_probable(echantillon)
    for r in resultats:
        assert all(score >= 0 for score in r["detail"].values())


def test_cause_probable_une_seule_ligne(explicateur, echantillon):
    ligne_unique = echantillon.iloc[[0]]
    resultats = explicateur.cause_probable(ligne_unique)
    assert len(resultats) == 1
    assert isinstance(resultats[0]["cause_principale"], str)


def test_familles_features_couvrent_les_features_actives():
    """Chaque feature listée dans une famille doit exister dans le dataset actif, sinon la famille est mal définie."""
    config = charger_config()
    colonnes_actives = set(config["jeu_donnees"]["features"])
    for famille, colonnes in FAMILLES_FEATURES.items():
        for col in colonnes:
            assert col in colonnes_actives, f"{col} (famille '{famille}') absente des features actives"


def test_ecart_thermique_couvre_bien_les_features_de_surchauffe():
    assert "ecart_thermique_max_24h" in FAMILLES_FEATURES["Surchauffe"]
