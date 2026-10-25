import pytest

from src.usine import CHAINE, cout_heure_arret, cout_panne, criticite, nom_machine, sante, statut, zone


def test_chaine_couvre_les_50_machines_une_seule_fois():
    numeros = [n for etape in CHAINE for n in etape["ids"]]
    assert sorted(numeros) == list(range(1, 51))


def test_criticite_inversement_proportionnelle_a_la_redondance():
    assert criticite("GDIZ_TEX_M_20") == 10.0  # 1 cardeuse sur 4
    assert criticite("GDIZ_TEX_M_30") == 5.0  # 1 rameuse sur 8
    assert criticite("GDIZ_TEX_M_23") == 1.7  # 1 métier à tisser sur 24
    assert nom_machine("GDIZ_TEX_M_20") == "Cardeuse"


def test_cout_panne_croit_avec_la_duree_et_la_gamme():
    assert cout_panne("M", 3) - cout_panne("M", 1) == pytest.approx(2 * cout_heure_arret("M"))
    assert cout_panne("H", 2) > cout_panne("M", 2) > cout_panne("L", 2)


def test_sante_et_statut():
    assert sante(0) == 100
    assert sante(0.298) == 24
    assert statut("eleve", "marche") == "panne_predite"
    assert statut("normal", "panne") == "en_panne"


@pytest.mark.parametrize(
    "risque,crit,attendu",
    [
        (0.01, 10, None),
        (0.03, 1.7, "surveiller"),
        (0.03, 5, "planifier"),
        (0.03, 10, "planifier"),
        (0.3, 1.7, "agir"),
        (0.3, 10, "agir"),
    ],
)
def test_zone(risque, crit, attendu):
    assert zone(risque, 0.029, crit) == attendu
