from streamlit.testing.v1 import AppTest


def test_dashboard_demarre_sans_erreur():
    """Exécute tout dashboard.py (3 onglets) : attrape les imports cassés et les requêtes invalides."""
    app = AppTest.from_file("dashboard.py", default_timeout=120).run()
    assert not app.exception
    assert app.metric, "aucun indicateur affiché"
