# Résumé de notre avancée

Architecture du Projet : Nous avons structuré le code selon les standards du marché (app/ pour le service, src/ pour la logique, data/ pour les futurs datasets).

Contrat de Données (Schemas) : Utilisation de Pydantic pour définir et valider les entrées de l'API (température, vitesse, etc.). Cela garantit que l'API ne plante pas si les données sont malformées.

Logique Métier & Feature Engineering : Création d'une fonction dans utils.py qui transforme les données brutes (ex: calcul du delta de température) avant de passer par un modèle heuristique.

API de Production (FastAPI) : Mise en place d'une API web capable de recevoir des requêtes JSON et de renvoyer des prédictions en temps réel avec une documentation interactive (Swagger).

Conteneurisation (Docker) : Création d'un Dockerfile pour empaqueter l'application et ses dépendances dans une image isolée et reproductible.