"""
Lance tout le projet sans Docker et ouvre le tableau de bord : python -m src.demarrer (make demo).
Registre MLflow local si aucun ne répond, données, entraînement et scoring seulement s'ils manquent. Ctrl+C arrête tout.
Avec --preparer : seulement les données, l'entraînement et le scoring manquants (service pipeline de docker-compose).
"""

import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser

import duckdb
import mlflow

from src.config import ALIAS_CHAMPION, CHEMIN_DB, NOM_MODELE, RACINE, TABLE_PREDICTIONS, URI_MLFLOW

URL_TABLEAU_DE_BORD = "http://localhost:8000"


def _repond(url: str) -> bool:
    try:
        return bool(urllib.request.urlopen(f"{url}/health", timeout=1))
    except OSError:
        return False


def _lancer(message: str, module: str) -> None:
    print(message, flush=True)
    subprocess.run([sys.executable, "-m", module], check=True, cwd=RACINE)


def _champion_existe() -> bool:
    mlflow.set_tracking_uri(URI_MLFLOW)
    try:
        mlflow.MlflowClient().get_model_version_by_alias(NOM_MODELE, ALIAS_CHAMPION)
        return True
    except Exception:
        return False


def _predictions_existent() -> bool:
    with duckdb.connect(str(CHEMIN_DB), read_only=True) as conn:
        return bool(conn.execute(f"SELECT 1 FROM duckdb_tables() WHERE table_name = '{TABLE_PREDICTIONS}'").fetchall())


def _demarrer_registre() -> subprocess.Popen | None:
    """Même serveur que docker-compose, mais local (mlflow.db et mlartifacts/ dans le projet) ; aucun si l'un répond."""
    if not URI_MLFLOW.startswith("http") or _repond(URI_MLFLOW):
        return None
    print(f"Démarrage du registre MLflow sur {URI_MLFLOW}…", flush=True)
    port = URI_MLFLOW.rsplit(":", 1)[-1]
    serveur = subprocess.Popen(
        [sys.executable, "-m", "mlflow", "server", "--backend-store-uri", "sqlite:///mlflow.db",
         "--artifacts-destination", "mlartifacts", "--port", port],
        cwd=RACINE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )  # fmt: skip
    for _ in range(60):
        if _repond(URI_MLFLOW):
            return serveur
        time.sleep(1)
    serveur.terminate()
    raise RuntimeError(f"Le registre MLflow ne répond pas sur {URI_MLFLOW}.")


def preparer() -> None:
    """Données, entraînement et scoring, chacun seulement s'il manque."""
    if not CHEMIN_DB.exists():
        _lancer("Génération des données (téléchargement de la météo au premier lancement)…", "simulation.generer")
    entraine = not _champion_existe()
    if entraine:
        _lancer("Entraînement du modèle…", "src.train")
    if entraine or not _predictions_existent():
        _lancer("Scoring des 60 derniers jours…", "src.modele")


def main():
    if "--preparer" in sys.argv:
        preparer()
        return
    serveur = _demarrer_registre()
    try:
        preparer()
        print(f"Tableau de bord sur {URL_TABLEAU_DE_BORD} (Ctrl+C pour tout arrêter)", flush=True)
        threading.Timer(3, webbrowser.open, [URL_TABLEAU_DE_BORD]).start()
        subprocess.run([sys.executable, "-m", "uvicorn", "app.main:app", "--port", "8000"], cwd=RACINE)
    except KeyboardInterrupt:
        pass
    finally:
        if serveur:
            serveur.terminate()


if __name__ == "__main__":
    main()
