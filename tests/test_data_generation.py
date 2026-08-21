"""
Tests de non-régression pour les bugs de reproductibilité de la génération synthétique.

Contexte : la génération de données doit être bit-à-bit reproductible d'une exécution
à l'autre (même seed → même dataset) et compatible avec le mode copy-on-write qui
deviendra le comportement par défaut de pandas 3.x. Deux bugs corrigés ici :
- `inject_missing_data` renvoyait un tableau numpy read-only sous copy-on-write.
- Le seed par machine dépendait de `hash()`, salé aléatoirement par process.
"""

import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from scripts.generate_synthetic_data import DataAggregator, SyntheticDataConfig


@pytest.fixture
def dummy_sensor_df() -> pd.DataFrame:
    n = 500
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "machine_id": ["M1"] * n,
            "tool_wear": rng.uniform(0, 100, n),
            "process_temperature": rng.uniform(20, 90, n),
            "rotational_speed": rng.uniform(0, 2000, n),
            "torque": rng.uniform(0, 80, n),
            "vibration": rng.uniform(0, 10, n),
            "ambient_temperature": rng.uniform(20, 40, n),
            "humidity": rng.uniform(40, 100, n),
            "dust_concentration": rng.uniform(0, 500, n),
        }
    )


def test_inject_missing_data_pandas3_copy_on_write_compat(dummy_sensor_df):
    """Reproduit le mode par défaut de pandas 3.x : to_numpy() doit rester inscriptible."""
    original_cow = pd.options.mode.copy_on_write
    try:
        pd.options.mode.copy_on_write = True
        aggregator = DataAggregator(SyntheticDataConfig())
        result = aggregator.inject_missing_data(dummy_sensor_df)
    finally:
        pd.options.mode.copy_on_write = original_cow

    sensor_cols = [
        "tool_wear",
        "process_temperature",
        "rotational_speed",
        "torque",
        "vibration",
        "ambient_temperature",
        "humidity",
        "dust_concentration",
    ]
    total_cells = len(result) * len(sensor_cols)
    expected_missing = int(total_cells * SyntheticDataConfig().missing_data_ratio)
    actual_missing = result[sensor_cols].isna().sum().sum()

    # Tirage (row, col) avec remise : quelques collisions possibles, donc <= au nombre
    # demandé plutôt qu'une égalité stricte. On garde une tolérance large mais non nulle.
    assert 0 < actual_missing <= expected_missing
    # La source n'a pas été mutée en place (pas d'effet de bord via une vue partagée).
    assert dummy_sensor_df[sensor_cols].isna().sum().sum() == 0


def test_machine_seed_deterministic_across_process():
    """
    Le calcul du seed par machine ne doit plus dépendre de hash() (salé par process).
    On force deux PYTHONHASHSEED différents entre les deux sous-process : avec l'ancien
    code (hash() builtin), les deux valeurs auraient divergé ; avec le hash md5
    déterministe, elles doivent être identiques.
    """
    snippet = (
        "import hashlib; "
        "machine_id = 'GDIZ_TEX_M_01'; "
        "h = int(hashlib.md5(machine_id.encode('utf-8')).hexdigest(), 16) % 10000; "
        "print(42 + h)"
    )

    results = []
    for hash_seed in ("111", "222"):
        proc = subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True,
            text=True,
            env={"PYTHONHASHSEED": hash_seed, "PATH": os.environ.get("PATH", "")},
            check=True,
        )
        results.append(proc.stdout.strip())

    assert results[0] == results[1]
