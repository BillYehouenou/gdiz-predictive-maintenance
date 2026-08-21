from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.train import ModelPipeline


@pytest.fixture
def pipeline():
    return ModelPipeline()


def test_temporal_split_no_leakage(pipeline):
    """Le split à 3 voies doit être strictement chronologique : train avant
    VALIDATION_SPLIT_DATE, validation entre les deux bornes, test à partir de
    SPLIT_DATE — sinon le modèle apprend sur le futur ou le seuil est choisi
    sur des données qui servent aussi à le rapporter."""
    pipeline.features_cols = ["sensor"]
    pipeline.target_col = "target"
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-06-01", "2024-11-01", "2024-12-15", "2025-01-01", "2025-06-01"],
            ),
            "sensor": [1.0, 2.0, 3.0, 4.0, 5.0],
            "target": [0, 1, 0, 1, 0],
        }
    )

    X_train, y_train, X_val, y_val, X_test, y_test = pipeline._temporal_split(df)

    assert len(X_train) == 1
    assert len(X_val) == 2
    assert len(X_test) == 2
    assert (pd.to_datetime(df.loc[X_train.index, "timestamp"]) < "2024-10-01").all()
    assert (
        (pd.to_datetime(df.loc[X_val.index, "timestamp"]) >= "2024-10-01")
        & (pd.to_datetime(df.loc[X_val.index, "timestamp"]) < "2025-01-01")
    ).all()
    assert (pd.to_datetime(df.loc[X_test.index, "timestamp"]) >= "2025-01-01").all()


def test_temporal_split_requires_timestamp_column(pipeline):
    df = pd.DataFrame({"sensor": [1.0, 2.0]})
    with pytest.raises(ValueError, match="timestamp"):
        pipeline._temporal_split(df)


def test_promote_if_better_auto_promotes_when_no_champion(pipeline):
    """Premier entraînement : aucun champion n'existe encore, la promotion doit être automatique."""
    pipeline.client = MagicMock()
    pipeline.client.get_model_version_by_alias.side_effect = Exception("RESOURCE_DOES_NOT_EXIST")

    promoted = pipeline._promote_if_better("ModelTest", new_version="1", new_score=0.10)

    assert promoted is True
    pipeline.client.set_registered_model_alias.assert_called_once_with("ModelTest", "champion", "1")


def test_promote_if_better_rejects_worse_score(pipeline):
    """Un nouveau modèle moins bon que le champion actuel ne doit jamais le remplacer."""
    pipeline.client = MagicMock()
    champion_version = MagicMock(run_id="run-abc", version="3")
    pipeline.client.get_model_version_by_alias.return_value = champion_version
    pipeline.client.get_run.return_value.data.metrics = {"recall_event": 0.50}

    promoted = pipeline._promote_if_better("ModelTest", new_version="4", new_score=0.20)

    assert promoted is False
    pipeline.client.set_registered_model_alias.assert_not_called()


def test_promote_if_better_promotes_when_strictly_better(pipeline):
    """Un nouveau modèle qui bat le champion actuel doit le remplacer."""
    pipeline.client = MagicMock()
    champion_version = MagicMock(run_id="run-abc", version="3")
    pipeline.client.get_model_version_by_alias.return_value = champion_version
    pipeline.client.get_run.return_value.data.metrics = {"recall_event": 0.30}

    promoted = pipeline._promote_if_better("ModelTest", new_version="4", new_score=0.45)

    assert promoted is True
    pipeline.client.set_registered_model_alias.assert_called_once_with("ModelTest", "champion", "4")
