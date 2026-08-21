import pandas as pd
import pytest

import src.preprocessor as preprocessor_module
from src.preprocessor import Preprocessor


@pytest.fixture
def configured_preprocessor(monkeypatch):
    """Preprocessor pointant vers une config minimale, sans toucher à la config réelle du projet."""
    fake_config = {"dataset": {"features": ["num1", "benin_season", "machine_type"]}}
    monkeypatch.setattr(preprocessor_module, "load_config", lambda: fake_config)
    return Preprocessor()


@pytest.fixture
def train_df():
    return pd.DataFrame(
        {
            "num1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "benin_season": [0, 1, 2, 3, 0, 1],
            "machine_type": ["L", "M", "H", "L", "M", "H"],
        }
    )


def test_numeric_features_not_scaled(configured_preprocessor, train_df):
    """Pas de StandardScaler (point 21) : les valeurs numériques restent inchangées (hors imputation)."""
    result = configured_preprocessor.fit_transform(train_df)
    assert result["num__num1"].tolist() == train_df["num1"].tolist()


def test_categorical_features_use_category_dtype(configured_preprocessor, train_df):
    """Encodage natif LightGBM (point 21) : les catégorielles sont typées 'category', pas one-hot."""
    result = configured_preprocessor.fit_transform(train_df)
    assert result["cat__benin_season"].dtype.name == "category"
    assert result["cat__machine_type"].dtype.name == "category"
    # Pas de colonnes one-hot du type "cat__machine_type_H"
    assert not any(c.startswith("cat__machine_type_") for c in result.columns)


def test_categorical_codes_stable_across_batches_with_fewer_categories(configured_preprocessor, train_df):
    """
    Un lot de transform() qui ne contient qu'une seule catégorie ne doit pas être
    ré-indexé différemment du fit — sinon LightGBM interprète mal les splits sur ce
    lot (cf. l'incident documenté dans docs/model-decisions.md).
    """
    configured_preprocessor.fit(train_df)

    full_batch = configured_preprocessor.transform(train_df)
    single_row = train_df.iloc[[2]]  # machine_type="H", benin_season=2
    single_result = configured_preprocessor.transform(single_row)

    expected_code = full_batch.loc[train_df["machine_type"] == "H", "cat__machine_type"].iloc[0]
    assert int(single_result["cat__machine_type"].iloc[0]) == int(expected_code)
    assert single_result["cat__machine_type"].dtype == full_batch["cat__machine_type"].dtype


def test_transform_before_fit_raises(configured_preprocessor, train_df):
    with pytest.raises(ValueError, match="fitted"):
        configured_preprocessor.transform(train_df)
