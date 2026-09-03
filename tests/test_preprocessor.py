import pandas as pd
import pytest

import src.preprocessor as preprocessor_module
from src.preprocessor import Preprocesseur


@pytest.fixture
def configured_preprocessor(monkeypatch):
    """Preprocesseur pointant vers une config minimale, sans toucher à la config réelle du projet."""
    fake_config = {"jeu_donnees": {"features": ["num1", "saison_benin", "type_machine"]}}
    monkeypatch.setattr(preprocessor_module, "charger_config", lambda: fake_config)
    return Preprocesseur()


@pytest.fixture
def train_df():
    return pd.DataFrame(
        {
            "num1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "saison_benin": [0, 1, 2, 3, 0, 1],
            "type_machine": ["L", "M", "H", "L", "M", "H"],
        }
    )


def test_features_categorielles_type_category(configured_preprocessor, train_df):
    result = configured_preprocessor.fit_transform(train_df)
    assert result["cat__saison_benin"].dtype.name == "category"
    assert result["cat__type_machine"].dtype.name == "category"
    # Pas de colonnes one-hot du type "cat__type_machine_H"
    assert not any(c.startswith("cat__type_machine_") for c in result.columns)


def test_codes_categoriels_stables_entre_lots_avec_moins_de_categories(configured_preprocessor, train_df):
    """
    Un lot de transform() qui ne contient qu'une seule catégorie ne doit pas être
    ré-indexé différemment du fit. Sinon LightGBM interprète mal les splits sur ce
    lot.
    """
    configured_preprocessor.fit(train_df)

    full_batch = configured_preprocessor.transform(train_df)
    single_row = train_df.iloc[[2]]  # type_machine="H", saison_benin=2
    single_result = configured_preprocessor.transform(single_row)

    expected_code = full_batch.loc[train_df["type_machine"] == "H", "cat__type_machine"].iloc[0]
    assert int(single_result["cat__type_machine"].iloc[0]) == int(expected_code)
    assert single_result["cat__type_machine"].dtype == full_batch["cat__type_machine"].dtype


def test_transform_avant_fit_leve_erreur(configured_preprocessor, train_df):
    with pytest.raises(ValueError, match="fitted"):
        configured_preprocessor.transform(train_df)
