import numpy as np
import pandas as pd
import pytest

from simulation.generer import COLONNES_CAPTEURS, defaillances_capteurs
from simulation.meteo import au_pas
from simulation.physique import ConfigDonneesSynthetiques, EvolueurEtatMachine, GenerateurContexteExogene


@pytest.fixture
def df_capteur_factice() -> pd.DataFrame:
    n = 3000
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "id_machine": ["M1"] * n,
            "temperature_process": rng.uniform(20, 90, n),
            "vitesse_rotation": rng.uniform(0, 2000, n),
            "couple": rng.uniform(0, 80, n),
            "vibration": rng.uniform(0, 10, n),
            "temperature_ambiante": rng.uniform(20, 40, n),
            "humidite": rng.uniform(40, 100, n),
            "concentration_poussiere": rng.uniform(0, 500, n),
        }
    )


@pytest.fixture
def config_courte() -> ConfigDonneesSynthetiques:
    config = ConfigDonneesSynthetiques()
    config.duree_jours = 60
    return config


@pytest.fixture
def meteo_horaire_factice(config_courte) -> pd.DataFrame:
    """Météo horaire plausible (cycle jour/nuit, épisode d'Harmattan), sans appel réseau."""
    index = pd.date_range(config_courte.date_debut, periods=config_courte.duree_jours * 24, freq="h")
    heures = index.hour.to_numpy()
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "temperature_exterieure": 27 + 4 * np.sin(2 * np.pi * (heures - 9) / 24),
            "humidite_exterieure": 80 - 10 * np.sin(2 * np.pi * (heures - 9) / 24),
            "precipitation_mm_h": np.where(rng.random(len(index)) < 0.03, 4.0, 0.0),
            "poussiere_exterieure": np.where(index.day < 10, 400.0, 10.0),
        },
        index=index,
    )


def test_injecter_donnees_manquantes_compat_pandas3_copy_on_write(df_capteur_factice):
    original_cow = pd.options.mode.copy_on_write
    try:
        pd.options.mode.copy_on_write = True
        result = defaillances_capteurs(df_capteur_factice, ConfigDonneesSynthetiques())
    finally:
        pd.options.mode.copy_on_write = original_cow

    # Sur 3000 pas, au moins une rafale de perte de paquets est quasi certaine
    assert result[COLONNES_CAPTEURS].isna().sum().sum() > 0
    assert df_capteur_factice[COLONNES_CAPTEURS].isna().sum().sum() == 0


def test_injecter_donnees_manquantes_produit_rafales_correlees(df_capteur_factice):
    """Les valeurs manquantes doivent former des rafales corrélées dans le temps."""
    result = defaillances_capteurs(df_capteur_factice, ConfigDonneesSynthetiques())

    is_nan = result["vibration"].isna().to_numpy().astype(int)
    assert is_nan.sum() > 0, "aucune rafale générée sur cet essai ; augmenter la taille de l'échantillon"

    diff = np.diff(np.concatenate([[0], is_nan, [0]]))
    run_lengths = np.flatnonzero(diff == -1) - np.flatnonzero(diff == 1)

    # La majorité des rafales doivent durer plus d'un pas
    assert (run_lengths > 1).mean() > 0.5


def test_au_pas_conserve_le_cumul_de_pluie_et_interpole(meteo_horaire_factice):
    timestamps = pd.date_range(meteo_horaire_factice.index[0], meteo_horaire_factice.index[-1], freq="30min")
    meteo = au_pas(meteo_horaire_factice, timestamps)

    assert not meteo.isna().any().any()
    # Le cumul est réparti sur les sous-pas, jamais dupliqué (hors première heure, cumul antérieur au début)
    assert meteo["precipitation_mm"].sum() == pytest.approx(meteo_horaire_factice["precipitation_mm_h"].iloc[1:].sum())
    # Un demi-pas tombe entre ses deux heures voisines
    t0, t1 = meteo_horaire_factice["temperature_exterieure"].iloc[:2]
    assert meteo["temperature_exterieure"].iloc[1] == pytest.approx((t0 + t1) / 2)


def test_contexte_atelier_suit_la_meteo_reelle(config_courte, meteo_horaire_factice):
    contexte = GenerateurContexteExogene(config_courte).generer_contexte_global(meteo_horaire_factice)

    assert len(contexte) == config_courte.duree_jours * 48
    # L'atelier est plus chaud que l'extérieur, donc plus sec en humidité relative
    assert contexte["temperature_ambiante"].mean() > meteo_horaire_factice["temperature_exterieure"].mean()
    assert contexte["humidite"].mean() < meteo_horaire_factice["humidite_exterieure"].mean()
    # L'épisode d'Harmattan (jours 1-9) se retrouve à l'intérieur
    harmattan = contexte["timestamp"].dt.day < 10
    assert (
        contexte.loc[harmattan, "concentration_poussiere"].mean() > 3 * contexte.loc[~harmattan, "concentration_poussiere"].mean()
    )


def test_etat_cache_jamais_exporte(config_courte, meteo_horaire_factice):
    """L'usure, l'encrassement et les roulements pilotent les pannes mais ne sont pas mesurables : jamais dans la télémétrie."""
    contexte = GenerateurContexteExogene(config_courte).generer_contexte_global(meteo_horaire_factice)
    df = EvolueurEtatMachine("GDIZ_TEX_M_01", "L", contexte, config_courte).evoluer_etats()

    colonnes = set(df.columns) | set(contexte.columns)
    assert not colonnes & {"usure_outil", "encrassement", "usure_roulements"}
    assert {"heures_depuis_changement_outil", "heures_depuis_maintenance"} <= colonnes


def test_compteur_outil_remis_a_zero_apres_panne_usure(config_courte, meteo_horaire_factice):
    contexte = GenerateurContexteExogene(config_courte).generer_contexte_global(meteo_horaire_factice)
    # Usure de départ juste sous le seuil personnel : une panne TWF arrive vite
    evolueur = EvolueurEtatMachine("GDIZ_TEX_M_01", "L", contexte, config_courte)
    evolueur.usure_outil[0] = evolueur.seuil_use
    df = evolueur.evoluer_etats()

    index_twf = np.flatnonzero(df["mode_panne"] == "TWF")
    assert len(index_twf) > 0
    heures_apres = df["heures_depuis_changement_outil"].iloc[index_twf[0] + 1 :]
    assert heures_apres.min() == 0.0


def test_etat_suit_les_arrets(config_courte, meteo_horaire_factice):
    contexte = GenerateurContexteExogene(config_courte).generer_contexte_global(meteo_horaire_factice)
    df = EvolueurEtatMachine("GDIZ_TEX_M_01", "L", contexte, config_courte).evoluer_etats()
    assert set(df["etat"]) <= {"marche", "arret", "delestage", "panne", "entretien"}
    assert (df.loc[df["panne"] == 1, "etat"] == "panne").all()
    marche = df.iloc[1:]  # la ligne 0 est l'état initial
    assert (marche.loc[marche["etat"] == "marche", "vitesse_rotation"] > 0).all()
