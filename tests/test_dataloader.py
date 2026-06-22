import pandas as pd
import pytest

import src.dataloader as dataloader_module
from src.dataloader import DataLoader


@pytest.fixture
def configured_loader(tmp_path, monkeypatch):
    """DataLoader pointant vers un répertoire temporaire, sans toucher à la config réelle du projet."""
    fake_config = {
        "paths": {"raw_data_path": str(tmp_path)},
        "dataset": {"name": "dataset.parquet", "target_column": "target_h120_predictable", "features": []},
    }
    monkeypatch.setattr(dataloader_module, "load_config", lambda: fake_config)
    return DataLoader()


def test_load_raw_data_missing_file(configured_loader):
    with pytest.raises(FileNotFoundError):
        configured_loader.load_raw_data()


def test_load_raw_data_unsupported_format(tmp_path, configured_loader):
    configured_loader.config["dataset"]["name"] = "dataset.txt"
    (tmp_path / "dataset.txt").write_text("contenu non supporté")
    with pytest.raises(ValueError, match="Format non supporté"):
        configured_loader.load_raw_data()


def test_load_raw_data_empty_returns_without_enrichment(tmp_path, configured_loader):
    pd.DataFrame().to_parquet(tmp_path / "dataset.parquet")
    df = configured_loader.load_raw_data()
    assert df.empty


def test_load_raw_data_parquet_enrichit_features_temporelles(tmp_path, configured_loader):
    """Vérifie que load_raw_data() applique bien enrich_training_df (colonnes temporelles ajoutées)."""
    raw = pd.DataFrame(
        {
            "machine_id": ["M1", "M1", "M1"],
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="30min"),
            "tool_wear": [10.0, 12.0, 14.0],
            "vibration": [1.0, 1.1, 1.2],
            "process_temperature": [40.0, 41.0, 42.0],
            "target": [0, 0, 0],
            "failure_mode": [None, None, None],
        }
    )
    raw.to_parquet(tmp_path / "dataset.parquet")

    df = configured_loader.load_raw_data()

    assert "tool_wear_delta_24h" in df.columns
    assert "vibration_max_24h" in df.columns
    assert len(df) == 3
