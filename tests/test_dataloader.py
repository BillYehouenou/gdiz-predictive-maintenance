import duckdb
import pandas as pd
import pytest

import src.dataloader as dataloader_module
from src.dataloader import ChargeurDonnees


@pytest.fixture
def configured_loader(tmp_path, monkeypatch):
    """ChargeurDonnees pointant vers une base DuckDB temporaire, sans toucher à la config réelle du projet."""
    fake_config = {
        "chemins": {"chemin_donnees_brutes": str(tmp_path)},
        "jeu_donnees": {
            "base": "jeu_donnees.db",
            "table": "telemetrie",
            "colonne_cible": "cible_h120_predictible",
            "features": [],
        },
    }
    monkeypatch.setattr(dataloader_module, "charger_config", lambda: fake_config)
    return ChargeurDonnees()


def _ecrire_table(tmp_path, df: pd.DataFrame):
    with duckdb.connect(str(tmp_path / "jeu_donnees.db")) as conn:
        conn.register("df", df)
        conn.execute("CREATE TABLE telemetrie AS SELECT * FROM df")


def test_charger_donnees_brutes_base_manquante(configured_loader):
    with pytest.raises(FileNotFoundError):
        configured_loader.charger_donnees_brutes()


def test_charger_donnees_brutes_vide_retourne_sans_enrichissement(tmp_path, configured_loader):
    _ecrire_table(tmp_path, pd.DataFrame({"id_machine": pd.Series(dtype=str), "vibration": pd.Series(dtype=float)}))
    df = configured_loader.charger_donnees_brutes()
    assert df.empty


def test_charger_donnees_brutes_enrichit_features_temporelles(tmp_path, configured_loader):
    """Vérifie que charger_donnees_brutes() applique bien enrichir_df_entrainement (colonnes temporelles ajoutées)."""
    raw = pd.DataFrame(
        {
            "id_machine": ["M1", "M1", "M1"],
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="30min"),
            "vibration": [1.0, 1.1, 1.2],
            "temperature_process": [40.0, 41.0, 42.0],
            "temperature_ambiante": [26.0, 26.5, 27.0],
            "concentration_poussiere": [30.0, 32.0, 35.0],
            "couple": [20.0, 22.0, 21.0],
            "panne": [0, 0, 0],
            "mode_panne": ["", "", ""],
        }
    )
    _ecrire_table(tmp_path, raw)

    df = configured_loader.charger_donnees_brutes()

    assert "vibration_max_24h" in df.columns
    assert "ecart_thermique" in df.columns
    assert "ecart_thermique_max_24h" in df.columns
    assert "poussiere_moyenne_24h" in df.columns
    assert "couple_max_24h" in df.columns
    assert len(df) == 3
