from unittest.mock import MagicMock

from src.config import NOM_MODELE
from src.train import promouvoir_si_meilleur


def _client(score_champion: float | None) -> MagicMock:
    client = MagicMock()
    if score_champion is None:
        client.get_model_version_by_alias.side_effect = Exception("RESOURCE_DOES_NOT_EXIST")
    else:
        client.get_run.return_value.data.metrics = {"recall_evenementiel": score_champion}
    return client


def test_promotion_automatique_sans_champion():
    client = _client(None)
    assert promouvoir_si_meilleur(client, "1", 0.10)
    client.set_registered_model_alias.assert_called_once_with(NOM_MODELE, "champion", "1")


def test_champion_conserve_si_score_inferieur():
    client = _client(0.50)
    assert not promouvoir_si_meilleur(client, "4", 0.20)
    client.set_registered_model_alias.assert_not_called()


def test_promotion_si_meilleur():
    client = _client(0.80)
    assert promouvoir_si_meilleur(client, "4", 0.95)
    client.set_registered_model_alias.assert_called_once_with(NOM_MODELE, "champion", "4")
