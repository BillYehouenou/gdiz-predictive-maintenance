from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.train import PipelineModele


@pytest.fixture
def pipeline():
    return PipelineModele()


def test_split_temporel_sans_fuite(pipeline):
    """Le split à 3 voies doit être strictement chronologique : train avant
    DATE_COUPURE_VALIDATION, validation entre les deux bornes, test à partir de
    DATE_COUPURE — sinon le modèle apprend sur le futur ou le seuil est choisi
    sur des données qui servent aussi à le rapporter."""
    pipeline.colonnes_features = ["capteur"]
    pipeline.colonne_cible = "panne"
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-06-01", "2024-11-01", "2024-12-15", "2025-01-01", "2025-06-01"],
            ),
            "capteur": [1.0, 2.0, 3.0, 4.0, 5.0],
            "panne": [0, 1, 0, 1, 0],
        }
    )

    X_train, y_train, X_val, y_val, X_test, y_test = pipeline._split_temporel(df)

    assert len(X_train) == 1
    assert len(X_val) == 2
    assert len(X_test) == 2
    assert (pd.to_datetime(df.loc[X_train.index, "timestamp"]) < "2024-10-01").all()
    assert (
        (pd.to_datetime(df.loc[X_val.index, "timestamp"]) >= "2024-10-01")
        & (pd.to_datetime(df.loc[X_val.index, "timestamp"]) < "2025-01-01")
    ).all()
    assert (pd.to_datetime(df.loc[X_test.index, "timestamp"]) >= "2025-01-01").all()


def test_split_temporel_requiert_colonne_timestamp(pipeline):
    df = pd.DataFrame({"capteur": [1.0, 2.0]})
    with pytest.raises(ValueError, match="timestamp"):
        pipeline._split_temporel(df)


def test_promouvoir_si_meilleur_promeut_auto_si_pas_de_champion(pipeline):
    """Premier entraînement : aucun champion n'existe encore, la promotion doit être automatique."""
    pipeline.client = MagicMock()
    pipeline.client.get_model_version_by_alias.side_effect = Exception("RESOURCE_DOES_NOT_EXIST")

    promu = pipeline._promouvoir_si_meilleur("ModelTest", nouvelle_version="1", nouveau_score=0.10)

    assert promu is True
    pipeline.client.set_registered_model_alias.assert_called_once_with("ModelTest", "champion", "1")


def test_promouvoir_si_meilleur_rejette_score_pire(pipeline):
    """Un nouveau modèle moins bon que le champion actuel ne doit jamais le remplacer."""
    pipeline.client = MagicMock()
    champion_version = MagicMock(run_id="run-abc", version="3")
    pipeline.client.get_model_version_by_alias.return_value = champion_version
    pipeline.client.get_run.return_value.data.metrics = {"recall_evenementiel": 0.50}

    promu = pipeline._promouvoir_si_meilleur("ModelTest", nouvelle_version="4", nouveau_score=0.20)

    assert promu is False
    pipeline.client.set_registered_model_alias.assert_not_called()


def test_promouvoir_si_meilleur_promeut_si_strictement_meilleur(pipeline):
    """Un nouveau modèle qui bat le champion actuel doit le remplacer."""
    pipeline.client = MagicMock()
    champion_version = MagicMock(run_id="run-abc", version="3")
    pipeline.client.get_model_version_by_alias.return_value = champion_version
    pipeline.client.get_run.return_value.data.metrics = {"recall_evenementiel": 0.30}

    promu = pipeline._promouvoir_si_meilleur("ModelTest", nouvelle_version="4", nouveau_score=0.45)

    assert promu is True
    pipeline.client.set_registered_model_alias.assert_called_once_with("ModelTest", "champion", "4")
