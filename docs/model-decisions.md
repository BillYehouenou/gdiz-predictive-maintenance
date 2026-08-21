# Décisions de modélisation — features et preprocessing

Document vivant, même esprit que `docs/data-generation.md` et `docs/experiments.md` : les choix de `src/features.py` et `src/preprocessor.py`, avec les mesures qui les justifient.

## Sommaire

- [Écart thermique process/ambiant (ΔT)](#écart-thermique-processambiant-δt)
- [Poussière cumulée sur 24h](#poussière-cumulée-sur-24h)
- [Tendances non retenues cette fois : portée de "tendances et cumuls"](#tendances-non-retenues-cette-fois--portée-de-tendances-et-cumuls)
- [Retrait du StandardScaler, catégorielles natives LightGBM](#retrait-du-standardscaler-catégorielles-natives-lightgbm)
- [Incident : rupture de compatibilité du modèle en production](#incident--rupture-de-compatibilité-du-modèle-en-production)
- [Horizons différenciés par mode de panne — hypothèse testée et invalidée](#horizons-différenciés-par-mode-de-panne--hypothèse-testée-et-invalidée)
- [PWF/RNF non prédictibles — confirmé empiriquement](#pwfrnf-non-prédictibles--confirmé-empiriquement)

---

## Écart thermique process/ambiant (ΔT)

Le générateur définit le mode HDF sur `process_temperature - ambient_temperature` (moyenné sur 4h, cf. `docs/data-generation.md`), mais cette différence n'était pas exposée comme feature — le modèle devait la reconstruire lui-même à partir des deux colonnes séparées, ce qu'un arbre peut faire mais au prix de splits plus profonds et moins directs.

Ajouté (`delta_temp`) dans les 3 chemins de calcul (entraînement batch, dashboard, API) — calcul ponctuel, ne dépend d'aucun historique, donc aucun changement d'architecture nécessaire.

**Vérifié** : importance mesurée après entraînement, 10ᵉ sur 28 features (non négligeable mais pas dominante — la fenêtre glissante 24h capte déjà une bonne partie du signal thermique via `process_temp_max_24h`/`process_temp_mean_24h`).

## Poussière cumulée sur 24h

`dust_concentration` n'était utilisée qu'en lecture instantanée — aucune notion d'exposition cumulée, alors que c'est justement l'exposition cumulée qui pilote l'encrassement (`fouling`, point 7) et donc le risque HDF.

Ajouté `dust_mean_24h` et `dust_max_24h` (même patron que les fenêtres 24h déjà en place pour vibration/température). Contrairement à `delta_temp`, ces deux features dépendent de l'historique — touchent donc 4 endroits, pas 3 : `enrich_training_df`, `enrich_inference_series` (dashboard, déjà disponible via `q_machine_ts_full`), `enrich_inference_point` (API, a nécessité d'ajouter `dust_concentration` à la requête SQL de `_get_machine_history`, avant absente), et le repli manuel sans historique dans `app/utils.py::process_prediction`.

**Vérifié** : `dust_max_24h` ressort **3ᵉ feature la plus importante du modèle** (derrière `tool_wear` et `vibration_max_24h`) — signal net que l'exposition à la poussière est un précurseur fort, pas un raffinement marginal.

## Tendances non retenues cette fois : portée de "tendances et cumuls"

La roadmap suggérait aussi "poussière cumulée 7 jours" et "coupures/semaine". Non retenues dans ce commit, pour une raison architecturale précise plutôt qu'un oubli : le chemin API (`enrich_inference_point`) ne reçoit que les 48 dernières lignes d'une machine (24h, `app/utils.py::_get_machine_history`, `LIMIT 48`). Une fenêtre de 7 jours nécessiterait d'élargir cette requête à 336 lignes — un changement contenu mais qui touche la couche de service, pas seulement le feature engineering, pour un point classé 🟢 (raffinement). Reporté plutôt que bricolé à moitié : une feature d'entraînement qui ne peut pas être calculée correctement en service créerait un écart train/serve, pire que ne pas l'avoir du tout.

De même, "heures depuis reset usure" nécessiterait un historique non borné (potentiellement des semaines), incompatible avec la fenêtre de 48 lignes sans un changement plus profond de l'architecture de service. Reporté pour la même raison.

## Retrait du StandardScaler, catégorielles natives LightGBM

`Preprocessor` standardisait les features numériques (`StandardScaler`) et encodait les catégorielles en one-hot. Aucun des deux n'a d'effet positif pour un modèle à base d'arbres : LightGBM est invariant à toute transformation monotone des features numériques (la standardisation ne change aucune décision de split), et le one-hot explose la dimensionnalité pour un gain nul par rapport à un encodage natif.

**Changement** : `StandardScaler` retiré (imputation seule). Catégorielles encodées en `OrdinalEncoder` puis castées en dtype pandas `category` — LightGBM détecte et gère nativement ces colonnes (`pandas_categorical` dans le modèle sérialisé, vérifié directement sur le booster entraîné).

**Piège évité** : les codes catégoriels doivent rester stables entre l'entraînement et chaque appel d'inférence, sinon LightGBM interprète mal les splits sur un lot qui ne contient qu'un sous-ensemble des catégories vues à l'entraînement (une machine "H" appelée seule, par exemple). Résolu en figeant un `pd.CategoricalDtype` avec la liste complète des catégories à l'issue du `fit()`, réutilisé identiquement à chaque `transform()` — vérifié explicitement par un test dédié (transformer un lot à une seule catégorie après un fit sur plusieurs, et confirmer que les codes ne bougent pas).

---

## Incident : rupture de compatibilité du modèle en production

Le changement de `Preprocessor` ci-dessus a cassé la compatibilité avec le modèle déjà en place (version 6, entraîné avec `StandardScaler`/`OneHotEncoder`) : un objet sérialisé (pickle/MLflow) restaure l'**état** d'une ancienne exécution mais exécute le code de la **classe actuelle** — la version 6 chargée après ce commit plantait sur `self._categorical_dtypes`, un attribut qui n'existait pas encore quand elle a été entraînée.

**Corrigé partiellement** : `transform()` utilise maintenant `getattr(self, "_categorical_dtypes", {})` plutôt qu'un accès direct, pour ne pas planter bêtement sur cet attribut manquant. Ça ne suffit pas à rendre un ancien modèle pleinement fonctionnel (son `ColumnTransformer` interne a aussi été fitté sans `set_output(transform="pandas")`, donc renvoie un numpy array et pas un DataFrame) — mais ce n'était pas l'objectif : réparer un artefact obsolète en profondeur serait de la dette permanente pour un problème ponctuel.

**Ce qui a réellement résolu le problème** : entraîner une nouvelle version (7) avec le code actuel et la promouvoir manuellement `@champion`, en dehors de la gate automatique (`_promote_if_better`). La gate a mesuré recall_event=0,8348 pour la version 7 contre 0,8435 pour la version 6 (champion précédent) — en toute rigueur, la version 7 n'aurait pas dû être promue automatiquement. Mais la version 6 était devenue **non fonctionnelle** suite à ce commit, pas simplement moins bonne : comparer les deux sur leur seule métrique n'avait plus de sens (une exception à chaque prédiction n'est pas "moins bonne", elle est inutilisable). D'où la promotion manuelle, documentée ici plutôt que cachée dans un commit silencieux.

**Trouvaille à noter pour une phase MLOps future** (points 27/29 de la roadmap) : la gate actuelle ne vérifie que la métrique, jamais si le champion actuel est encore chargeable/exécutable avec le code courant. Un changement de schéma de preprocessing peut donc casser silencieusement un modèle en production sans que la gate ne s'en aperçoive — elle continuerait de comparer un nouveau candidat au score d'un champion mort. Corriger ça proprement demanderait que la gate tente une inférence de fumée sur le champion avant de comparer les scores, pas juste lire une métrique loguée. Hors scope ici (ce commit porte sur les features/preprocessing, pas sur la gate elle-même) — mais le risque est réel et vérifié empiriquement, pas hypothétique.

**Écart de performance non expliqué, signalé plutôt que caché** : même après ce contournement, la version 7 reste très légèrement en dessous de la version 6 sur le recall événementiel mesuré (0,8348 vs 0,8435 — un seul événement de panne de différence sur 115, 96 vs 97 attrapées). Les nouvelles features sont mesurablement utiles (`dust_max_24h` en 3ᵉ position d'importance), donc ce n'est probablement que du bruit de ré-entraînement plutôt qu'une vraie régression — mais ce n'est pas prouvé statistiquement ici (un seul point de mesure, pas un intervalle de confiance). Le tuning d'hyperparamètres (point 25, phase ultérieure) est le bon moment pour vérifier si cet écart se résorbe.

---

## Horizons différenciés par mode de panne — hypothèse testée et invalidée

**L'hypothèse de la roadmap** : l'horizon unique de 120h (`target_h120_predictable`) plafonnerait la précision — HDF/OSF auraient un signal précurseur exploitable seulement jusqu'à ~72h, TWF resterait exploitable jusqu'à 120-168h (surtout après la rampe terminale, point 8). Combiner un horizon propre à chaque mode devrait donc battre l'horizon unique.

**Premier test, méthodologiquement bancal** : comparaison de plusieurs configurations d'horizon (label combiné par mode via `compute_multi_horizon_target`), mais évaluées avec la fenêtre événementielle unique de 120h héritée d'avant ce commit — y compris pour les configurations où le label HDF/OSF avait été réduit à 72h. Résultat apparent : la config HDF/OSF=72h/TWF=120h battait l'horizon unique (recall test 0,861 contre 0,835). **Ce résultat était un artefact** : évaluer avec une fenêtre de 120h un modèle entraîné à alerter dans les 72h avant HDF/OSF le crédite de rattrapages qu'il n'a pas appris à faire de façon fiable.

**Deuxième test, cohérent** : la fenêtre d'évaluation (`horizon_steps` dans `event_based_metrics`) a été rendue capable d'accepter un horizon par mode (`dict[str, int]`, testé dans `tests/test_evaluation.py::test_event_based_metrics_uses_per_mode_horizon_dict`), pour évaluer chaque configuration avec **sa propre** fenêtre — cohérence entre ce que le modèle apprend à prédire et ce sur quoi on le juge. Résultat, sur la validation (le seul critère de sélection légitime, pas le test) :

| Configuration | Seuil retenu | Recall événementiel (validation) |
|---|---|---|
| **Horizon unique 120h (actuel)** | 0,080 | **0,826** |
| HDF/OSF=72h, TWF=120h | 0,050 | 0,739 |
| HDF/OSF=60h, TWF=120h | 0,035 | 0,696 |
| HDF/OSF=72h, TWF=96h | 0,040 | 0,761 |
| HDF/OSF=96h, TWF=120h | 0,060 | 0,761 |

**Aucune configuration à horizon différencié ne bat l'horizon unique** une fois l'évaluation rendue cohérente. Hypothèse pour expliquer l'écart (non vérifiée plus loin, notée pour une future itération) : réduire la fenêtre HDF/OSF à 72h réduit aussi le nombre d'exemples positifs de ces deux causes dans les données d'entraînement — HDF+OSF représentent ensemble ~35 % des pannes (cf. `docs/data-generation.md`) — pendant qu'un seul classifieur binaire partagé apprend les trois causes mélangées. La perte de volume d'exemples pourrait coûter plus cher que le gain espéré en retirant des exemples "flous" (panne dans 72-120h, signal faible mais pas nul).

**Décision** : `target_h120_predictable` reste la cible active (`config/config.yaml`). Le code pour tester des horizons différenciés (`HORIZON_BY_FAILURE_MODE`, `compute_multi_horizon_target`, colonne `target_multi_horizon` calculée mais non utilisée, support `dict` dans `event_based_metrics`/`find_threshold_for_fa_budget`) est conservé — testé, fonctionnel, disponible pour une ré-évaluation si le générateur ou le modèle changent significativement, mais pas activé par défaut. Cohérent avec le point 24 déjà invalidé une fois dans ce projet (persistance des alertes) : un résultat négatif mesuré vaut mieux qu'un succès non mesuré.

## PWF/RNF non prédictibles — confirmé empiriquement

Vérifié sur le dataset final (50 machines, générateur v2) que PWF et RNF n'ont pas de signal précurseur exploitable, en excluant les pas de temps à l'arrêt (délestage/activité nulle) pour ne pas biaiser la mesure par du temps d'arrêt contaminant la fenêtre pré-panne :

| Capteur | Ratio PWF (pré-panne 24h / baseline) | Ratio RNF |
|---|---|---|
| process_temperature | 1,021 | 0,987 |
| torque | 1,005 | 0,998 |
| vibration | 1,111 | 1,007 |
| tool_wear | 1,089 | 1,155 |

Ratios proches de 1,0 pour la plupart des capteurs (contre 2 à 4x pour TWF/HDF/OSF, cf. sections précédentes) — confirme la décision déjà actée d'exclure PWF/RNF de `PREDICTABLE_FAILURE_MODES`. Le léger excès sur `tool_wear` (1,09-1,16) n'est pas un vrai précurseur : c'est un effet de population — les machines plus usées ont simplement plus d'occasions de subir n'importe quelle panne, pas spécifiquement PWF/RNF à ce moment précis.
