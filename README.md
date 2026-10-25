# GDIZ Maintenance prédictive

Anticiper les pannes de machines textiles avant qu'elles n'arrêtent la production. Un atelier textile béninois, sur 2 ans avec des données de capteurs IoT, méteo et réseau SBEE. Chaque machine est évaluée toutes les 30 minutes pour répondre à une question : **va-t-elle tomber en panne dans les 5 prochains jours et pourquoi ?** 


![Démonstration du tableau de bord : indicateurs, plan de l'atelier, matrice risque × criticité, fiche machine](docs/demo.gif)


## Essayer en une commande

Prérequis : Python 3.11+ et [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/BillYehouenou/gdiz-predictive-maintenance.git
cd gdiz-predictive-maintenance
make install
make demo
```

Ou, avec Docker seulement (ni Python ni uv) : `docker compose up`, puis http://localhost:8000.


## Déploiement Docker

La stack Compose enchaîne trois services : le registre MLflow (port 5050), un pipeline qui génère les données,
entraîne et score le modèle s'ils manquent, puis l'API avec le tableau de bord (port 8000), qui charge le @champion
et son seuil depuis le registre. Données et registre sont gardés dans des volumes : environ 2 min 30 au premier
lancement, une quinzaine de secondes ensuite.

```bash
make docker-up    # démarre tout et rend la main quand le tableau de bord est prêt
make docker-down  # arrête (docker compose down -v efface aussi les données et le registre)
```

*GitHub Actions* : lint + mypy ; tests avec génération des données et entraînement à la volée ; build Docker. S'exécute sur chaque push et PR vers main.


## Sous le capot

- **Des données crédibles** : météo réelle Open-Meteo, physique d'usure et de surchauffe, capteurs IoT qui
  décrochent ou dérivent ; l'état réel des machines reste caché, seuls les capteurs et la GMAO sont visibles.
- **Une métrique métier soutenable** : part des pannes annoncées dans les 5 jours (et avec au moins 4 h de préavis),
  sous un budget de fausses alertes comptées en interventions, avec la précision des alertes et le bilan en FCFA.
- **Un modèle honnête** : LightGBM comparé à trois autres modèles (régression logistique, arbre, HistGradientBoosting), appris sur les seules
  machines en marche, probabilités calibrées, cause probable regroupée par famille physique.
- **Du MLOps adéquat** : registre MLflow (le modèle n'est promu que si son gain net en validation ne baisse pas, avec
  son seuil), suivi de la performance réelle des 60 derniers jours et de la dérive à saison égale (`make check-drift`),
  DuckDB embarqué, API FastAPI, tableau de bord HTML/CSS/JS sans framework, tests, CI et Docker.



## Choix du modèle

Part des pannes prévisibles (usure, surchauffe, surcharge) annoncées dans les 5 jours, au budget de 0,2 fausse alerte
par machine et par jour ([notebooks/1_comparaison_modeles.ipynb](notebooks/1_comparaison_modeles.ipynb), qui
détaille aussi les décisions de modélisation) :

| Modèle | Validation 2024 | Test 2025 (115 pannes) | dont ≥ 4 h de préavis | Précision des alertes | Gain net 2025 |
|---|---|---|---|---|---|
| Régression logistique | 72 % | 47 % | 41 % | 2,0 % | −3,5 M FCFA |
| Arbre de décision | 69 % | 53 % | 44 % | 1,4 % | −2,5 M FCFA |
| HistGradientBoosting | 77 % | 63 % | 54 % | 1,7 % | 0,0 M FCFA |
| **LightGBM (déployé)** | 74 % | **64 %** | **55 %** | 1,9 % | −0,6 M FCFA |

Incertitude : ±5 à 6 points de recall. Le boosting annonce le plus de pannes, et LightGBM est conservé, plus rapide
et plus léger que HistGradientBoosting dont il ne se distingue pas. En FCFA, le bilan reste proche de zéro : le gain net
compte une panne évitée comme un arrêt planifié de 2 h sans rebut, et une fausse alerte comme une heure de technicien
(`src/usine.py`).


## Limites

1. **Données synthétiques, pas des capteurs GDIZ réels** : le générateur reproduit des dynamiques physiques
   plausibles (calendrier saisonnier béninois, réseau SBEE, Harmattan), mais reste une simulation calibrée.
2. **La surcharge mécanique (OSF) reste le point faible**, faute d'exemples ; les pannes électriques et aléatoires
   n'ont pas de signe avant-coureur et ne sont pas apprises.
3. **Une alerte sur cinquante seulement précède une panne.** Le modèle n'est rentable que si une inspection coûte
   moins d'environ 3 100 FCFA (moins d'une heure de technicien). Piste : un seuil plus haut, choisi sur une
   validation plus longue.
4. **Préavis médian d'environ 12 h**, pas plusieurs jours : 55 % des pannes prévisibles sont annoncées au moins 4 h à
   l'avance, assez pour planifier une intervention le jour même, pas pour commander une pièce.
5. **La cause probable est une famille physique (SHAP)** — usure, surchauffe, surcharge, environnement — et non un
   diagnostic du mode de panne : un classifieur multi-classe testé à part était trop instable pour être exposé.

