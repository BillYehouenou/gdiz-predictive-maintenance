from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_pages_servies():
    for chemin, script in [("/", "parc.js"), ("/machines/GDIZ_TEX_M_20", "machine.js")]:
        r = client.get(chemin)
        assert r.status_code == 200
        assert f"/static/js/{script}" in r.text


def test_page_machine_inconnue():
    assert client.get("/machines/inconnue").status_code == 404


def test_machines_redirige_vers_la_plus_a_risque():
    r = client.get("/machines", follow_redirects=False)
    assert r.status_code == 307
    assert r.headers["location"].startswith("/machines/GDIZ_TEX_M_")


def test_fichiers_statiques():
    for fichier in ["css/tokens.css", "css/app.css", "js/api.js", "js/charts.js", "js/nav.js"]:
        assert client.get(f"/static/{fichier}").status_code == 200
