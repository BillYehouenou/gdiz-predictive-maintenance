# Expériences et mesures — méthodologie d'évaluation

Document vivant, même esprit que `docs/data-generation.md` : les choix de `src/train.py` et `src/evaluation.py`, avec les mesures qui les justifient. Chiffres mesurés sur le dataset final (50 machines, générateur v2 — cf. `docs/data-generation.md`), pas recopiés d'une itération précédente.

## Sommaire

- [Pourquoi remplacer le F2 par-pas de temps](#pourquoi-remplacer-le-f2-par-pas-de-temps)
- [Split à 3 voies : train / validation / test](#split-à-3-voies--train--validation--test)
- [Courbe recall / fausses alertes et choix du budget](#courbe-recall--fausses-alertes-et-choix-du-budget)
- [Résultat final sur le test](#résultat-final-sur-le-test)
- [Angle mort découvert : l'alerte permanente notée comme parfaite](#angle-mort-découvert--lalerte-permanente-notée-comme-parfaite)
- [Calibration isotonique des probabilités](#calibration-isotonique-des-probabilités)
- [Baselines de référence](#baselines-de-référence)

**Note sur les chiffres ci-dessous** : les sections "Courbe recall/fausses alertes" et "Résultat final" datent d'avant la correction du délai de réarmement (section suivante) — les tableaux y sont corrects pour le code de l'époque, mais désormais dépassés par le fix. Chiffres à jour dans la section "Angle mort découvert" et au-delà.

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

---

## Angle mort découvert : l'alerte permanente notée comme parfaite

**Comment il a été trouvé** : en implémentant la calibration isotonique (point 22, section suivante), le seuil optimal retenu s'est mis à tomber systématiquement sur 0,005 — la valeur plancher de la grille de recherche. Signal d'alarme : un optimum qui colle au bord de sa grille de recherche n'est presque jamais un vrai optimum.

**Ce qui se passait réellement** : à ce seuil, le modèle calibré était en alerte **69,6 % du temps** (jusqu'à 100 % avec une calibration sigmoïde) — un système qui reste "allumé" en continu pendant des mois n'est d'aucune utilité opérationnelle. Pourtant, la métrique événementielle telle que construite en séance précédente (comptage par épisode, sans limite de durée) notait ça comme quasi parfait : `false_alarm_rate_per_machine_day` = 0,058, parce qu'une alerte ininterrompue de 300 jours ne compte que comme **un seul** épisode, peu importe sa durée.

C'est exactement l'inverse du problème que le comptage par épisode devait résoudre (une alerte continue de 10h ne doit pas compter comme 20 fausses alertes) — poussé à l'extrême, il permet à un modèle inutilisable d'obtenir un score presque nul en fausses alertes, simplement en ne s'arrêtant jamais.

**Le fix** : délai de réarmement (`DEFAULT_MAX_EPISODE_STEPS = 48` pas = 24h dans `src/evaluation.py`) — tout épisode continu plus long que 24h est redécoupé en plusieurs épisodes. C'est le raffinement que la doc de méthodologie originale du projet avait déjà anticipé sans le juger nécessaire à l'époque ("délai de réarmement après intervention technicien") — il l'était.

**Vérifié** (`tests/test_evaluation.py`) : un modèle en alerte permanente sur 10 jours donne maintenant 10 épisodes (1/jour) et un taux de FA ≈ 1,0/machine/jour — plus 0. Un run continu de 10 pas avec `max_episode_steps=4` se redécoupe bien en `[(0,3), (4,7), (8,9)]`.

**Conséquence sur les chiffres déjà publiés** : le tableau de la section "Courbe recall/fausses alertes" ci-dessus (seuil 0,085, mesuré avant ce fix) reste correct pour le code de l'époque, mais les seuils optimaux recalculés après le fix diffèrent — cf. section suivante pour les chiffres à jour. Pas de quoi remettre en cause le principe du comptage par épisode (point 24 reste juste), seulement sa version initiale, incomplète sur les cas extrêmes.

---

## Calibration isotonique des probabilités

**Le problème** : `Predictor._normalize_probability` (ancien `src/xpredict.py`) recentrait les probabilités brutes sur un axe logit pour que le seuil de décision s'affiche artificiellement à 50 % — un ravalement d'affichage, pas une vraie calibration. Le reste de la courbe (au-dessus/en dessous de 50 %) restait aussi mal calibré qu'avant : une probabilité affichée à 70 % ne voulait rien dire de particulier sur la fréquence réelle de panne parmi les cas similaires.

**Le choix** : `CalibratedClassifierCV(FrozenEstimator(full_pipeline), method="isotonic")`, ajustée sur la validation (le pipeline est déjà entraîné sur train — seule la calibration doit être fittée, `FrozenEstimator` l'exprime explicitement). Le pipeline calibré est celui qui est sauvegardé/loggé — `predict_proba()` dessus renvoie directement des probabilités interprétables, plus besoin de recentrage côté service.

**Vérifié — trois méthodes comparées, une fois le délai de réarmement corrigé** (seuil choisi sur validation, rapporté sur test, jamais l'inverse) :

| Méthode | Seuil retenu | Recall (validation) | Recall (test) | FA/machine/jour (test) | Brier (test) |
|---|---|---|---|---|---|
| Brute (pas de calibration) | 0,140 | 71,7 % | 78,3 % | 0,197 | 0,0244 |
| **Isotonique (retenue)** | 0,120 | 71,7 % | **79,1 %** | 0,201 | **0,0243** |
| Sigmoïde (Platt) | 0,065 | 71,7 % | 78,3 % | 0,196 | 0,0244 |

Écarts modestes entre les trois — attendu : la calibration reformule les valeurs de probabilité, elle ne change pas le classement relatif des exemples produit par LightGBM, dont dépend directement le recall événementiel à seuil optimal. L'isotonique l'emporte de justesse sur le recall test et le Brier score, retenue pour ça et parce que c'était la méthode demandée. Le vrai gain n'est pas visible dans ce tableau : `failure_probability` est maintenant une probabilité au sens propre, pas un artefact d'affichage — vérifiable par le Brier score (erreur quadratique moyenne entre probabilité prédite et résultat réel), qui n'existait pas comme métrique de suivi avant ce commit.

**Risque assumé** : l'isotonique est non paramétrique et peut suradapter avec peu d'exemples positifs en calibration (46 pannes en validation) — c'est d'ailleurs ce qui a produit l'artefact (calibration en escalier avec seulement 193 valeurs distinctes sur 873 600 prédictions test). Le délai de réarmement (section précédente) protège contre la conséquence la plus grave (seuil dégénéré non détecté), mais la courbe de calibration elle-même reste plus instable qu'avec une méthode paramétrique (sigmoïde). À surveiller si le budget de fausses alertes est resserré dans une itération future.

**Conséquence directe traitée** : les seuils fixes de risque du dashboard (0,35 "Surveillance" / 0,50 "Risque élevé", `ui/tabs/tab_machine.py`) et du tag `alert_level` du monitoring MLflow (`app/utils.py`) supposaient un seuil de décision proche de 0,5 — plus vrai une fois la calibration réelle en place (seuil optimal mesuré : 0,12). Les deux ont été recalculés relativement à `predictor.threshold` plutôt que codés en dur, pour rester cohérents avec la décision réelle du modèle quel que soit le seuil retenu à un ré-entraînement futur.

---

## Baselines de référence

Point 23 : le modèle LightGBM doit battre une règle triviale et une régression logistique sur la métrique événementielle, pas juste exister. Mesuré via `scripts/run_baseline_comparison.py`, même infrastructure d'évaluation (seuil choisi sur validation, rapporté sur test) :

| Modèle | Seuil | Recall événementiel (test) | FA/machine/jour (test) | Pannes attrapées |
|---|---|---|---|---|
| Règle (`tool_wear > seuil`) | 64 % | 74,8 % | 0,171 | 86/115 |
| Régression logistique | 0,825 | 74,8 % | 0,165 | 86/115 |
| **LightGBM calibré (retenu)** | 0,12 | **79,1 %** | 0,201 | 91/115 |

Le modèle bat les deux baselines sur le recall événementiel (+4,3 points), au prix d'un taux de fausses alertes un peu plus élevé (les deux baselines restent en dessous du budget de 0,20 sans l'épuiser — elles plafonnent en recall avant d'atteindre la limite de FA autorisée, signe qu'usure et régression linéaire séparent moins bien les cas limites que LightGBM). Coïncidence notable : la règle triviale et la régression logistique attrapent exactement les mêmes 86 pannes — `tool_wear` seul explique une bonne partie du signal linéairement exploitable, la régression logistique ne fait pas mieux qu'un seuil brut sur cette tâche.

Marge réelle mais pas écrasante : confirme que LightGBM apporte une valeur mesurable (pas un choix de modèle arbitraire), sans prétendre à une supériorité spectaculaire que les chiffres ne montrent pas.
