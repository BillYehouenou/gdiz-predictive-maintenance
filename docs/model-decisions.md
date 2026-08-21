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
- [Pourquoi HDF/OSF sont moins bien détectés que TWF, et ce qui a été tenté](#pourquoi-hdfosf-sont-moins-bien-détectés-que-twf-et-ce-qui-a-été-tenté)
- [Ré-audit de la variable cible et de l'architecture — trois hypothèses testées, trois échecs](#ré-audit-de-la-variable-cible-et-de-larchitecture--trois-hypothèses-testées-trois-échecs)
- [Étage 2 : diagnostic de cause (point 14)](#étage-2--diagnostic-de-cause-point-14)
- [Fenêtres 72h/120h, pentes et ratio — trois idées testées, aucune retenue](#fenêtres-72h120h-pentes-et-ratio--trois-idées-testées-aucune-retenue)

---

## Écart thermique process/ambiant (ΔT)

Le générateur définit le mode HDF sur `temperature_process - temperature_ambiante` (moyenné sur 4h, cf. `docs/data-generation.md`), mais cette différence n'était pas exposée comme feature — le modèle devait la reconstruire lui-même à partir des deux colonnes séparées, ce qu'un arbre peut faire mais au prix de splits plus profonds et moins directs.

Ajouté (`ecart_thermique`) dans les 3 chemins de calcul (entraînement batch, dashboard, API) — calcul ponctuel, ne dépend d'aucun historique, donc aucun changement d'architecture nécessaire.

**Vérifié** : importance mesurée après entraînement, 10ᵉ sur 28 features (non négligeable mais pas dominante — la fenêtre glissante 24h capte déjà une bonne partie du signal thermique via `temperature_process_max_24h`/`temperature_process_moyenne_24h`).

## Poussière cumulée sur 24h

`concentration_poussiere` n'était utilisée qu'en lecture instantanée — aucune notion d'exposition cumulée, alors que c'est justement l'exposition cumulée qui pilote l'encrassement (`encrassement`, point 7) et donc le risque HDF.

Ajouté `poussiere_moyenne_24h` et `poussiere_max_24h` (même patron que les fenêtres 24h déjà en place pour vibration/température). Contrairement à `ecart_thermique`, ces deux features dépendent de l'historique — touchent donc 4 endroits, pas 3 : `enrichir_df_entrainement`, `enrichir_serie_inference` (dashboard, déjà disponible via `q_serie_complete_machine`), `enrichir_point_inference` (API, a nécessité d'ajouter `concentration_poussiere` à la requête SQL de `_recuperer_historique_machine`, avant absente), et le repli manuel sans historique dans `app/utils.py::traiter_prediction`.

**Vérifié** : `poussiere_max_24h` ressort **3ᵉ feature la plus importante du modèle** (derrière `usure_outil` et `vibration_max_24h`) — signal net que l'exposition à la poussière est un précurseur fort, pas un raffinement marginal.

## Tendances non retenues cette fois : portée de "tendances et cumuls"

La roadmap suggérait aussi "poussière cumulée 7 jours" et "coupures/semaine". Non retenues dans ce commit, pour une raison architecturale précise plutôt qu'un oubli : le chemin API (`enrichir_point_inference`) ne reçoit que les 48 dernières lignes d'une machine (24h, `app/utils.py::_recuperer_historique_machine`, `LIMIT 48`). Une fenêtre de 7 jours nécessiterait d'élargir cette requête à 336 lignes — un changement contenu mais qui touche la couche de service, pas seulement le feature engineering, pour un point classé 🟢 (raffinement). Reporté plutôt que bricolé à moitié : une feature d'entraînement qui ne peut pas être calculée correctement en service créerait un écart train/serve, pire que ne pas l'avoir du tout.

De même, "heures depuis reset usure" nécessiterait un historique non borné (potentiellement des semaines), incompatible avec la fenêtre de 48 lignes sans un changement plus profond de l'architecture de service. Reporté pour la même raison.

## Retrait du StandardScaler, catégorielles natives LightGBM

`Preprocesseur` standardisait les features numériques (`StandardScaler`) et encodait les catégorielles en one-hot. Aucun des deux n'a d'effet positif pour un modèle à base d'arbres : LightGBM est invariant à toute transformation monotone des features numériques (la standardisation ne change aucune décision de split), et le one-hot explose la dimensionnalité pour un gain nul par rapport à un encodage natif.

**Changement** : `StandardScaler` retiré (imputation seule). Catégorielles encodées en `OrdinalEncoder` puis castées en dtype pandas `category` — LightGBM détecte et gère nativement ces colonnes (`pandas_categorical` dans le modèle sérialisé, vérifié directement sur le booster entraîné).

**Piège évité** : les codes catégoriels doivent rester stables entre l'entraînement et chaque appel d'inférence, sinon LightGBM interprète mal les splits sur un lot qui ne contient qu'un sous-ensemble des catégories vues à l'entraînement (une machine "H" appelée seule, par exemple). Résolu en figeant un `pd.CategoricalDtype` avec la liste complète des catégories à l'issue du `fit()`, réutilisé identiquement à chaque `transform()` — vérifié explicitement par un test dédié (transformer un lot à une seule catégorie après un fit sur plusieurs, et confirmer que les codes ne bougent pas).

---

## Incident : rupture de compatibilité du modèle en production

Le changement de `Preprocesseur` ci-dessus a cassé la compatibilité avec le modèle déjà en place (version 6, entraîné avec `StandardScaler`/`OneHotEncoder`) : un objet sérialisé (pickle/MLflow) restaure l'**état** d'une ancienne exécution mais exécute le code de la **classe actuelle** — la version 6 chargée après ce commit plantait sur `self._types_categoriels`, un attribut qui n'existait pas encore quand elle a été entraînée.

**Corrigé partiellement** : `transform()` utilise maintenant `getattr(self, "_types_categoriels", {})` plutôt qu'un accès direct, pour ne pas planter bêtement sur cet attribut manquant. Ça ne suffit pas à rendre un ancien modèle pleinement fonctionnel (son `ColumnTransformer` interne a aussi été fitté sans `set_output(transform="pandas")`, donc renvoie un numpy array et pas un DataFrame) — mais ce n'était pas l'objectif : réparer un artefact obsolète en profondeur serait de la dette permanente pour un problème ponctuel.

**Ce qui a réellement résolu le problème** : entraîner une nouvelle version (7) avec le code actuel et la promouvoir manuellement `@champion`, en dehors de la gate automatique (`_promouvoir_si_meilleur`). La gate a mesuré recall_event=0,8348 pour la version 7 contre 0,8435 pour la version 6 (champion précédent) — en toute rigueur, la version 7 n'aurait pas dû être promue automatiquement. Mais la version 6 était devenue **non fonctionnelle** suite à ce commit, pas simplement moins bonne : comparer les deux sur leur seule métrique n'avait plus de sens (une exception à chaque prédiction n'est pas "moins bonne", elle est inutilisable). D'où la promotion manuelle, documentée ici plutôt que cachée dans un commit silencieux.

**Trouvaille à noter pour une phase MLOps future** (points 27/29 de la roadmap) : la gate actuelle ne vérifie que la métrique, jamais si le champion actuel est encore chargeable/exécutable avec le code courant. Un changement de schéma de preprocessing peut donc casser silencieusement un modèle en production sans que la gate ne s'en aperçoive — elle continuerait de comparer un nouveau candidat au score d'un champion mort. Corriger ça proprement demanderait que la gate tente une inférence de fumée sur le champion avant de comparer les scores, pas juste lire une métrique loguée. Hors scope ici (ce commit porte sur les features/preprocessing, pas sur la gate elle-même) — mais le risque est réel et vérifié empiriquement, pas hypothétique.

**Écart de performance non expliqué, signalé plutôt que caché** : même après ce contournement, la version 7 reste très légèrement en dessous de la version 6 sur le recall événementiel mesuré (0,8348 vs 0,8435 — un seul événement de panne de différence sur 115, 96 vs 97 attrapées). Les nouvelles features sont mesurablement utiles (`poussiere_max_24h` en 3ᵉ position d'importance), donc ce n'est probablement que du bruit de ré-entraînement plutôt qu'une vraie régression — mais ce n'est pas prouvé statistiquement ici (un seul point de mesure, pas un intervalle de confiance). Le tuning d'hyperparamètres (point 25, phase ultérieure) est le bon moment pour vérifier si cet écart se résorbe.

---

## Horizons différenciés par mode de panne — hypothèse testée et invalidée

**L'hypothèse de la roadmap** : l'horizon unique de 120h (`cible_h120_predictible`) plafonnerait la précision — HDF/OSF auraient un signal précurseur exploitable seulement jusqu'à ~72h, TWF resterait exploitable jusqu'à 120-168h (surtout après la rampe terminale, point 8). Combiner un horizon propre à chaque mode devrait donc battre l'horizon unique.

**Premier test, méthodologiquement bancal** : comparaison de plusieurs configurations d'horizon (label combiné par mode via `calculer_cible_multi_horizon`), mais évaluées avec la fenêtre événementielle unique de 120h héritée d'avant ce commit — y compris pour les configurations où le label HDF/OSF avait été réduit à 72h. Résultat apparent : la config HDF/OSF=72h/TWF=120h battait l'horizon unique (recall test 0,861 contre 0,835). **Ce résultat était un artefact** : évaluer avec une fenêtre de 120h un modèle entraîné à alerter dans les 72h avant HDF/OSF le crédite de rattrapages qu'il n'a pas appris à faire de façon fiable.

**Deuxième test, cohérent** : la fenêtre d'évaluation (`pas_horizon` dans `metriques_evenementielles`) a été rendue capable d'accepter un horizon par mode (`dict[str, int]`, testé dans `tests/test_evaluation.py::test_metriques_evenementielles_utilise_dict_horizon_par_mode`), pour évaluer chaque configuration avec **sa propre** fenêtre — cohérence entre ce que le modèle apprend à prédire et ce sur quoi on le juge. Résultat, sur la validation (le seul critère de sélection légitime, pas le test) :

| Configuration | Seuil retenu | Recall événementiel (validation) |
|---|---|---|
| **Horizon unique 120h (actuel)** | 0,080 | **0,826** |
| HDF/OSF=72h, TWF=120h | 0,050 | 0,739 |
| HDF/OSF=60h, TWF=120h | 0,035 | 0,696 |
| HDF/OSF=72h, TWF=96h | 0,040 | 0,761 |
| HDF/OSF=96h, TWF=120h | 0,060 | 0,761 |

**Aucune configuration à horizon différencié ne bat l'horizon unique** une fois l'évaluation rendue cohérente. Hypothèse pour expliquer l'écart (non vérifiée plus loin, notée pour une future itération) : réduire la fenêtre HDF/OSF à 72h réduit aussi le nombre d'exemples positifs de ces deux causes dans les données d'entraînement — HDF+OSF représentent ensemble ~35 % des pannes (cf. `docs/data-generation.md`) — pendant qu'un seul classifieur binaire partagé apprend les trois causes mélangées. La perte de volume d'exemples pourrait coûter plus cher que le gain espéré en retirant des exemples "flous" (panne dans 72-120h, signal faible mais pas nul).

**Décision** : `cible_h120_predictible` reste la cible active (`config/config.yaml`). Le code pour tester des horizons différenciés (`HORIZON_PAR_MODE_PANNE`, `calculer_cible_multi_horizon`, colonne `cible_multi_horizon` calculée mais non utilisée, support `dict` dans `metriques_evenementielles`/`trouver_seuil_pour_budget_fa`) est conservé — testé, fonctionnel, disponible pour une ré-évaluation si le générateur ou le modèle changent significativement, mais pas activé par défaut. Cohérent avec le point 24 déjà invalidé une fois dans ce projet (persistance des alertes) : un résultat négatif mesuré vaut mieux qu'un succès non mesuré.

## PWF/RNF non prédictibles — confirmé empiriquement

Vérifié sur le dataset final (50 machines, générateur v2) que PWF et RNF n'ont pas de signal précurseur exploitable, en excluant les pas de temps à l'arrêt (délestage/activité nulle) pour ne pas biaiser la mesure par du temps d'arrêt contaminant la fenêtre pré-panne :

| Capteur | Ratio PWF (pré-panne 24h / baseline) | Ratio RNF |
|---|---|---|
| temperature_process | 1,021 | 0,987 |
| couple | 1,005 | 0,998 |
| vibration | 1,111 | 1,007 |
| usure_outil | 1,089 | 1,155 |

Ratios proches de 1,0 pour la plupart des capteurs (contre 2 à 4x pour TWF/HDF/OSF, cf. sections précédentes) — confirme la décision déjà actée d'exclure PWF/RNF de `MODES_PANNE_PREDICTIBLES`. Le léger excès sur `usure_outil` (1,09-1,16) n'est pas un vrai précurseur : c'est un effet de population — les machines plus usées ont simplement plus d'occasions de subir n'importe quelle panne, pas spécifiquement PWF/RNF à ce moment précis.

---

## Pourquoi HDF/OSF sont moins bien détectés que TWF, et ce qui a été tenté

**Le constat de départ** (cf. `docs/experiments.md`, section "État actuel") : recall événementiel par mode, sur le champion d'alors — TWF 100 %, HDF 68 %, OSF 47 %. Le recall global (79 %) était tiré vers le haut par TWF, qui représente la moitié des pannes du test.

**Diagnostic — importances de features** : `usure_outil` domine à lui seul (524, plus du double de la 2ᵉ feature), alors que `couple` (le vrai moteur du hasard OSF dans le générateur) ne pèse que 9. Hypothèse : le modèle utilise `usure_outil` comme signal générique "quelque chose ne va pas" pour toutes les causes (une machine plus près de n'importe quelle panne a *en général* plus d'usure accumulée, par pur effet de calendrier), plutôt que d'apprendre le vrai signal spécifique à OSF.

**Vérification par signal auto-référencé** (moyenne par tranche de temps-avant-panne, comparée à une référence lointaine de la même population — méthode déjà utilisée pour PWF/RNF ci-dessus, pour éviter le biais de population) :

| Mode | Signal | Fenêtre où le signal apparaît | Amplitude relative |
|---|---|---|---|
| OSF | `couple` | dans les 24h précédant la panne seulement | +52 % vs référence lointaine (24,66 → 37,5) |
| OSF | `usure_outil` | présent mais faible sur tout l'horizon | +2,6 % seulement |
| HDF | `ecart_thermique` | dans les 24-48h précédant la panne | +64 % vs référence lointaine |

Conclusion : le signal OSF existe et est fort (+52 %), mais il est **court** (24h, pas plus — cohérent avec le point 15 déjà invalidé pour un horizon d'entraînement différencié) et **bruité** (couple instantané, qui varie à chaque pas avec l'activité). Le modèle n'avait accès qu'à cette version bruyante, jamais lissée.

**Ce qui a été tenté et mesuré** (comparaison sur validation avant tout ajout au code, cf. méthodologie habituelle) :

| Ajout testé | Recall validation | OSF (test) | HDF (test) | Retenu ? |
|---|---|---|---|---|
| (aucun, référence) | 0,717 | 47 % | 68 % | — |
| `couple_max_24h` seul | 0,739 | 57 % | 71 % | ✅ |
| `couple_max_12h` (fenêtre plus courte) | 0,761 | 53 % | 71 % | ❌ (FA test à 0,234, au-delà du budget) |
| `couple_x_usure` (interaction) | — | — | — | ❌ (pas dans le top 15 des importances, aucun apport mesuré) |
| `ecart_thermique_moyenne_4h` (vraie fenêtre du générateur) | 0,761 | 53 % | 79 % | ❌ (retenu concurrent ci-dessous à la place) |
| `ecart_thermique_max_24h` | 0,761 | 53 % | 75 % | ✅ |
| Les deux ensemble (`couple_moyenne_24h` + `max_24h`) | 0,696 | 47 % | 64 % | ❌ (redondance, dilue le signal) |

**Retenu** : `couple_max_24h` et `ecart_thermique_max_24h` — cohérents avec le patron déjà établi ailleurs dans le code (`vibration_max_24h`, `temperature_process_max_24h`, `poussiere_max_24h` existent déjà, pas besoin d'inventer une nouvelle convention de fenêtre). Implémentés dans les 4 chemins habituels (entraînement, dashboard, API avec/sans historique) — l'API a nécessité d'ajouter `couple` et `temperature_ambiante` à la requête d'historique DuckDB (`app/utils.py::_recuperer_historique_machine`), qui ne les chargeait pas avant.

**Résultat mesuré, pipeline réel (calibré, seuil 0,12)** :

| Mode | Avant | Après | Δ |
|---|---|---|---|
| Recall global | 79,1 % | **80,9 %** | +1,8 pt |
| TWF | 100 % | 100 % | — |
| HDF | 71,4 % | 71,4 % | — (le gain mesuré en isolation ne s'est pas confirmé une fois combiné à la calibration — cf. note ci-dessous) |
| OSF | 46,7 % | **53,3 %** | +6,6 pts |

**Écart entre le test isolé et le pipeline réel, signalé plutôt qu'ignoré** : `ecart_thermique_max_24h` seul montrait +7 pts sur HDF dans un test hors pipeline (modèle brut, pas calibré, seuil choisi différemment) ; une fois passé par la calibration isotonique et le seuil réellement retenu, ce gain HDF ne s'est pas matérialisé — HDF reste à 71,4 %. Pas d'explication tranchée ici (interaction entre la nouvelle feature et la courbe de calibration, possiblement) ; noté comme tel plutôt que de prétendre à un gain qui ne s'observe pas dans les chiffres qui comptent réellement (le pipeline de production, pas un script de test isolé).

**Ce qui reste un vrai point faible** : OSF, à 53 %, capture un peu plus d'une panne sur deux — mieux qu'avant, mais loin d'être résolu. Le signal physique existe (couple +52 % dans les 24h précédentes) mais reste intrinsèquement court et partiellement confondu avec le bruit d'activité normale. Deux pistes non tentées, discutées mais pas implémentées :

1. **Modifier le générateur** pour rendre le précurseur OSF plus long/moins bruité — par exemple une composante de charge "structurelle" qui monte progressivement avant l'OSF (au lieu du couple purement instantané piloté par l'activité), à la manière de la rampe terminale ajoutée pour TWF au point 8. Non fait ici : ça change la physique du générateur, donc oblige à revalider tout le travail de calibration des parts de pannes (`docs/data-generation.md`), potentiellement tout le pipeline de calibration isotonique et les budgets de FA en aval — un chantier bien plus large que l'ajout de deux features, à traiter comme son propre sujet plutôt que noyé dans celui-ci.
2. **Étage de classification dédié par cause** (point 14 de la roadmap) — testé depuis, cf. section suivante : n'améliore pas la détection, contrairement à l'intuition initiale.

---

## Ré-audit de la variable cible et de l'architecture — trois hypothèses testées, trois échecs

Suite à un doute légitime sur la cohérence de la cible active (`cible_h120_predictible`) après plusieurs séances de modifications successives (features, calibration, métrique), trois architectures alternatives ont été testées empiriquement plutôt que supposées meilleures ou pires.

**Ce qu'on détecte, pour mémoire** : pas "panne à cet instant", mais "cette ligne est-elle dans les 120h qui précèdent une panne TWF/HDF/OSF ?" — un état de risque soutenu sur une fenêtre de 5 jours, pas un instant de bascule. PWF/RNF exclus (point 18, aucun signal précurseur mesurable).

### Hypothèse 1 — classifieurs séparés par cause (au lieu d'un modèle mutualisé)

Intuition testée : un classifieur OSF dédié, non concurrencé par le signal TWF dominant (`usure_outil` pesait 524 en importance contre 9 pour `couple`), devrait mieux apprendre le signal OSF spécifique.

**Résultat : l'inverse.** Un classifieur OSF seul, quel que soit l'horizon (24h à 120h), plafonne à 43,3 % de recall test — pire que le modèle mutualisé (53,3 %). Cause identifiée : seulement **23 pannes OSF en train** (contre 115 pannes toutes causes confondues pour le modèle poolé) — pas assez d'exemples pour qu'un classifieur isolé apprenne quoi que ce soit de robuste. Le modèle mutualisé n'est pas un compromis bancal, c'est le choix qui a le plus de données pour apprendre le signal général de dégradation (usure, vibration) qui, même générique, reste plus utile à OSF qu'un signal OSF-only entraîné sur un échantillon trop petit. Confirme *a posteriori* la justification donnée par la roadmap elle-même pour l'étage 1 mutualisé ("robuste, tous positifs poolés") — c'était déjà la bonne réponse.

### Hypothèse 2 — horizons différenciés par mode, re-testés avec les nouvelles features

Invalidée une première fois avant l'ajout de `couple_max_24h`/`ecart_thermique_max_24h` (cf. section précédente). Re-testée après, au cas où les nouvelles features changeraient la conclusion :

| Configuration | Recall validation |
|---|---|
| **Horizon unique 120h (actuel)** | **0,739** |
| HDF/OSF=72h, TWF=120h | 0,696 |
| HDF/OSF=72h, TWF=168h | 0,630 |
| HDF/OSF=48h, TWF=120h | 0,630 |

Toujours invalidée, avec un effet secondaire notable : réduire la fenêtre OSF à 72h fait *chuter* le recall OSF lui-même (57 %→43 %) plutôt que l'améliorer — réduire la fenêtre réduit le nombre d'exemples positifs OSF disponibles pour le modèle mutualisé, ce qui coûte plus cher que le bruit qu'on retire n'en fait gagner.

### Hypothèse 3 — sur-pondérer les exemples OSF à l'entraînement

`sample_weight` sur les lignes en fenêtre pré-OSF, poids ×2 à ×5. Résultat : neutre au mieux (poids×1 = référence à 0,739/57 % de recall OSF), dégradé au pire (poids×2 → 0,717/50 %). Le modèle avec poids uniformes reste optimal.

### Conclusion du ré-audit

**La cible et l'architecture actuelles (`cible_h120_predictible`, horizon unique, modèle mutualisé) ne sont pas une incohérence — c'est, à ce stade, l'option la mieux vérifiée du projet.** Trois alternatives plausibles ont été testées avec la même rigueur que les décisions précédentes (mesure avant conclusion, jamais l'inverse), les trois ont perdu. Le vrai facteur limitant pour OSF n'est pas un choix de cible ou d'architecture mal aligné : c'est la rareté des exemples (23-30 pannes OSF selon le split) combinée à un signal physique intrinsèquement court (24h, cf. section précédente). Ni le label, ni le découpage du modèle ne peuvent compenser un manque de données — seuls plus de données (plus de machines, plus de temps de simulation) ou une refonte du signal physique du générateur le pourraient, et les deux sont des chantiers distincts, pas des corrections de cohérence.

---

## Étage 2 : diagnostic de cause (point 14)

Contrairement à l'intuition qui a motivé le ré-audit ci-dessus, le point 14 de la roadmap ne promettait pas d'améliorer la détection : l'étage 1 y est explicitement décrit comme "mutualisé, robuste" (confirmé ci-dessus), l'étage 2 est un **diagnostic**, entraîné uniquement sur les lignes déjà en fenêtre pré-panne, pour répondre à "laquelle des 3 causes est la plus probable" une fois l'alerte de l'étage 1 levée.

**Implémenté** : `src/features.py::calculer_label_cause_panne` (cible : la cause de la prochaine panne prédictible dans les 120h, None ailleurs) + `scripts/run_cause_classifier.py` (LightGBM multi-classe, `class_weight="balanced"`, entraîné sur train restreint aux lignes pré-panne).

**Mesuré** :

| | Validation | Test |
|---|---|---|
| Accuracy globale | 46,8 % | 64,4 % |
| TWF (recall) | — | 92 % |
| HDF (recall) | — | 44 % |
| OSF (recall) | 5 % | 32 % |

L'accuracy test (64,4 %) est dans l'ordre de grandeur cité par la roadmap ("~70 % à 30 machines"), mais deux réserves empêchent de le considérer comme fiable en l'état :

1. **Écart validation/test important** (46,8 % vs 64,4 %) — signe que la mesure elle-même est instable, cohérent avec le nombre réduit de pannes en validation (46 événements, comme pour toutes les métriques de ce projet).
2. **OSF quasi non diagnostiqué en validation** (5 % de recall — le classifieur ne le reconnaît presque jamais) — la même rareté d'exemples qui limite la détection (hypothèse 1 ci-dessus) limite aussi le diagnostic.

**Décision** : gardé comme script disponible et testé (`scripts/run_cause_classifier.py`), **pas intégré au dashboard/API**. L'intégrer maintenant afficherait une "cause probable" fiable une fois sur deux au mieux, et quasiment jamais pour OSF — ce serait pire que ne rien afficher. Point 33 (dashboard, phase ultérieure) est le bon moment pour décider comment présenter ce diagnostic avec ses limites, une fois qu'il aura gagné en fiabilité (plus de données) ou qu'une présentation honnête de l'incertitude aura été conçue.

---

## Fenêtres 72h/120h, pentes et ratio — trois idées testées, aucune retenue

Trois pistes de feature engineering ont été testées pour tenter d'améliorer la détection OSF/HDF au-delà du plateau documenté ci-dessus (OSF 53,3 %, HDF 71,4 %) : moyennes/écarts-types glissants à 72h et 120h (en plus du 24h déjà en place), pentes (vitesse d'accumulation) et un ratio vibration/couple censé isoler l'usure mécanique de la charge de production. Les trois ont été implémentées dans les 3 chemins de calcul habituels (`enrichir_df_entrainement`, `enrichir_serie_inference`, `enrichir_point_inference` — ce dernier a nécessité d'élargir l'historique chargé par l'API de 48 à 240 lignes, `app/utils.py::_recuperer_historique_machine`), puis mesurées avant d'être activées dans `config/config.yaml`.

**Résultat global (12 nouvelles features actives, seuil choisi sur validation)** :

| Configuration | Recall événementiel (test) | FA/machine/jour | TWF | HDF | OSF |
|---|---|---|---|---|---|
| **Champion actuel (14 features temporelles)** | **80,9 %** | 0,211 | 100 % | 71,4 % | **53,3 %** |
| + 12 features (72h/120h + pentes + ratio) | 77,4 % | 0,140 | 100 % | 71,4 % | 40,0 % |

Le gate de promotion (`_promouvoir_si_meilleur`) a refusé cette version — correctement, la baisse est réelle et mesurée sur le test, pas un artefact de seuil.

**Diagnostic — importances de features** : les moyennes 72h/120h (`poussiere_moyenne_120h`, `vibration_moyenne_120h`, `vibration_ecart_type_72h`) prennent les 3 premières places d'importance après `usure_outil`, au prix de `couple_max_24h` — le levier qui avait mesurablement amélioré OSF dans la section précédente (+10 pts) — dont l'importance chute de 152 à 75. Interprétation : ces nouvelles moyennes sont des versions plus lisses (donc plus faciles à découper pour l'arbre) du signal de dégradation général déjà couvert par les fenêtres 24h existantes — elles n'apportent pas d'information nouvelle sur OSF spécifiquement, mais consomment du budget de split que LightGBM aurait sinon alloué à `couple_max_24h`. Cohérent avec la conclusion déjà établie dans le ré-audit ci-dessus : **OSF est limité par la rareté des exemples (23-30 pannes), pas par un manque de features** — ajouter des features corrélées à un signal déjà présent dilue plutôt qu'il n'aide, quand le modèle n'a pas assez d'exemples positifs pour apprendre à les pondérer correctement.

**Ablation — isoler chaque idée** (retire les 9 moyennes 72h/120h, ne garde que pentes/ratio, sur la config 14+ratio+pentes) :

| Configuration | Recall événementiel (test) | OSF | Note |
|---|---|---|---|
| 14 features + ratio + pentes (sans les moyennes 72h/120h) | 80,0 % | 43,3 % | proche du champion en global, OSF toujours en retrait |
| 14 features + pentes seules (sans ratio) | 77,4 % | — | pas d'amélioration en retirant le ratio |

Le ratio vibration/couple ressort avec une **importance de 0 dans toutes les configurations testées** — LightGBM (modèle à base d'arbres) n'en a pas besoin : un arbre peut déjà apprendre la même interaction ("vibration élevée ET couple faible") via deux splits successifs sur les variables séparées, sans qu'un ratio pré-calculé n'apporte d'information supplémentaire. C'est une limite connue des ratios pour les modèles à base d'arbres (utiles surtout pour les modèles linéaires) — l'intuition physique derrière l'idée reste valide, mais LightGBM n'en tire aucun bénéfice mesurable.

`pente_vibration_72h` (vitesse d'accumulation du broutement) est la seule des 12 nouvelles features à ressortir avec une importance notable (108, dans le top 10) dans la config "ratio + pentes" — mais son effet net sur le recall global reste dans le bruit de mesure de ce projet (46 pannes en validation, 115 en test — cf. `docs/experiments.md`), pas un gain confirmé.

**Décision** : `config/config.yaml` reste sur les 14 features temporelles déjà validées — aucune des trois idées testées n'a battu le champion actuel sur le recall événementiel. Le code des 12 nouvelles features reste disponible dans `src/features.py` (`VARIABLES_TEMPORELLES`, fenêtres `FENETRE_72H`/`FENETRE_120H`) et fonctionnel de bout en bout (dashboard, API avec historique élargi à 120h) — testé, mesuré, gardé pour une ré-évaluation future si le volume de données OSF augmente, mais pas activé par défaut. Même logique que les horizons différenciés par mode (section "Horizons différenciés") : un résultat négatif mesuré vaut mieux qu'un ajout non vérifié.
