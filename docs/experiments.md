# Expériences et mesures — méthodologie d'évaluation

Document vivant, même esprit que `docs/data-generation.md` : les choix de `src/train.py` et `src/evaluation.py`, avec les mesures qui les justifient. Chiffres mesurés sur le dataset final (50 machines, générateur v2 — cf. `docs/data-generation.md`), pas recopiés d'une itération précédente.

## Sommaire

- [Pourquoi remplacer le F2 par-pas de temps](#pourquoi-remplacer-le-f2-par-pas-de-temps)
- [Split à 3 voies : train / validation / test](#split-à-3-voies--train--validation--test)
- [Courbe recall / fausses alertes et choix du budget](#courbe-recall--fausses-alertes-et-choix-du-budget)
- [Résultat final sur le test](#résultat-final-sur-le-test)

---

## Pourquoi remplacer le F2 par-pas de temps

Le système évalue chaque machine toutes les 30 min. Une machine qui reste en alerte 10h d'affilée comptait, dans l'ancien système, comme **20 fausses alertes** (20 pas à "alerte") — alors que pour l'équipe de maintenance, c'est un seul dérangement : on va voir la machine une fois.

**Mesuré sur le dataset final**, avec le seuil retenu (0,085) sur le test 2025 :

| Comptage | Taux |
|---|---|
| Par pas de 30 min (ancien) | 4,30 / machine / jour |
| Par épisode (nouveau) | 0,214 / machine / jour |

Rapport **×20,1** — dû uniquement à l'unité de mesure, sur les mêmes prédictions. "4,3 fausses alertes/machine/jour" (système inutilisable) et "environ 1 fausse alerte tous les 5 jours/machine" (défendable) décrivent exactement la même chose.

**Définition d'un épisode** (`src/evaluation.py::_alert_episodes`) : suite ininterrompue de pas en alerte sur une même machine.

Le F2-score par-pas est retiré de partout (seuil, gate de promotion `@champion`, rapport) — remplacé par deux chiffres qui se lisent sans traduction : le **recall événementiel** (part des pannes réellement précédées d'une alerte) et le **taux de fausses alertes par machine-jour**, toujours rapportés ensemble (l'un sans l'autre se triche trivialement : tout alerter = 100 % de recall mais système inutilisable ; ne jamais alerter = 0 fausse alerte mais tout raté).

---

## Split à 3 voies : train / validation / test

**Le problème** : jusqu'ici, le seuil de décision était choisi en optimisant une métrique **sur le test set lui-même** — puis ce même test set servait à rapporter la performance. Un seuil ajusté sur les données qu'il sert ensuite à évaluer est optimiste par construction, même si le modèle n'a jamais été entraîné dessus.

**Le choix** : `_temporal_split` découpe maintenant en 3 blocs chronologiques disjoints (`src/train.py`) :

| Bloc | Période | Rôle |
|---|---|---|
| Train | < 2024-10-01 | Entraînement du modèle |
| Validation | 2024-10-01 → 2025-01-01 | Choix du seuil (recall événementiel sous budget de FA) — jamais pour entraîner |
| Test | ≥ 2025-01-01 | Rapport final — jamais vu avant, ni pour entraîner, ni pour choisir le seuil |

Fenêtre de validation choisie (3 mois, oct-déc 2024) pour disposer d'assez de pannes pour une courbe recall/FA stable, sans trop rogner sur le train. Mesuré : train 1,05M lignes, validation 131k lignes (46 pannes prédictibles), test 613k lignes (115 pannes prédictibles).

---

## Courbe recall / fausses alertes et choix du budget

Balayage de seuils sur la **validation** (jamais sur le test), modèle final (50 machines, physique v2) :

| Seuil | Recall événementiel | FA / machine / jour | Pannes attrapées |
|---|---|---|---|
| 0,020 | 91,3 % | 0,384 | 42/46 |
| 0,040 | 84,8 % | 0,344 | 39/46 |
| 0,060 | 82,6 % | 0,278 | 38/46 |
| **0,085** | **76,1 %** | **0,198** | **35/46** |
| 0,100 | 76,1 % | 0,169 | 35/46 |
| 0,150 | 71,7 % | 0,125 | 33/46 |
| 0,200 | 56,5 % | 0,112 | 26/46 |
| 0,300 | 54,3 % | 0,100 | 25/46 |

**Budget retenu : 0,20 fausse alerte/machine/jour** (`FALSE_ALARM_BUDGET_PER_MACHINE_DAY` dans `src/train.py`) — repère "un dérangement pour rien tous les 5 jours par machine", jugé défendable pour une équipe de maintenance. Pas une valeur arbitraire copiée d'un exemple illustratif : elle correspond au coude visible de la courbe ci-dessus (entre 0,06 et 0,10, le recall décroche fort — de 82,6 % à 76,1 % puis stagne ; en dessous de 0,10, chaque point de FA en moins coûte cher en recall). `find_threshold_for_fa_budget` sélectionne automatiquement le seuil le plus généreux en recall qui respecte ce budget : **0,085**.

Cette valeur est un paramètre produit, pas un fait technique — modifiable en une ligne (`FALSE_ALARM_BUDGET_PER_MACHINE_DAY`), à ajuster si l'équipe terrain a un avis différent sur ce qui est acceptable.

---

## Résultat final sur le test

Seuil 0,085 (choisi sur validation), évalué sur le test 2025 (jamais vu avant) :

| Métrique | Validation | Test |
|---|---|---|
| Recall événementiel | 76,1 % | **84,3 %** |
| FA / machine / jour | 0,198 | **0,214** |
| Pannes attrapées | 35/46 | 97/115 |
| Préavis médian | — | **62 h** (~2,6 jours) |

**Écart validation → test à surveiller, pas caché** : le recall est meilleur sur test (84,3 % contre 76,1 %) mais le taux de FA dépasse légèrement le budget (0,214 contre la cible 0,20). Deux explications possibles, non tranchées ici : (a) la fenêtre de validation (oct-déc, 3 mois) ne couvre qu'une partie du cycle saisonnier — la petite saison sèche/Harmattan y est surreprésentée par rapport au test qui couvre une année complète ; (b) 46 pannes en validation contre 115 en test, l'intervalle de confiance sur le seuil optimal est large avec un échantillon aussi petit. Distinguer les deux demanderait une validation croisée temporelle (plusieurs fenêtres de validation), hors scope ici — noté comme piste pour une itération future plutôt que résolu par supposition.

PR-AUC (test) : 0,353 — retenue comme métrique secondaire de référence plutôt que ROC-AUC (0,859, toujours logué mais moins informatif ici : le taux de pannes réel sur le test est de 3,1 %, et le ROC-AUC est connu pour paraître artificiellement bon sous ce niveau de déséquilibre de classes).
