# Génération des données synthétiques — méthodologie et choix

Ce document explique **pourquoi** le générateur (`scripts/generate_synthetic_data.py`) est construit comme il l'est, et **ce qui a changé** au fil des itérations, avec la justification de chaque choix. C'est un document vivant : il est mis à jour à chaque commit qui touche la génération de données, pas rédigé après coup.

Statut des sections : les choix décrits ici sont des calibrations pour un générateur synthétique, pas des mesures de terrain. Quand un chiffre s'appuie sur une climatologie généralement documentée pour le Sud-Bénin plutôt que sur une donnée mesurée sur site GDIZ, c'est dit explicitement — pas de fausse précision.

## Sommaire

- [Calendrier saisonnier du Sud-Bénin](#calendrier-saisonnier-du-sud-bénin)
- [Réseau électrique](#réseau-électrique)
- [Physique des pannes](#physique-des-pannes)
- [Maintenance préventive imparfaite](#maintenance-préventive-imparfaite)
- [Données manquantes structurées](#données-manquantes-structurées)
- [Taille du parc : 50 machines](#taille-du-parc--50-machines)

---

## Calendrier saisonnier du Sud-Bénin

### Le problème

La première version du générateur modélisait le climat avec **une seule saison binaire** :

```python
rainy_months = [4, 5, 6, 7, 10]
seasons[np.isin(months, rainy_months)] = 0   # "saison des pluies"
# tout le reste (dont août, novembre-mars, septembre) = 1, "saison sèche", base_temp = 33.5°C
```

Deux problèmes concrets :

1. **Le Sud-Bénin n'a pas un cycle climatique unimodal.** Contrairement au Bénin septentrional (une seule grande saison des pluies), la bande côtière (climat guinéen, zone GDIZ comprise) a un cycle **bimodal** : deux saisons des pluies, deux saisons sèches. C'est dû à la remontée d'eau froide atlantique (upwelling) qui refroidit la façade côtière entre juillet et septembre. C'est une caractéristique climatologique couramment documentée pour cette région, pas une bizarrerie de calibration.
2. **Conséquence directe** : août — qui appartient à la *petite saison sèche*, la période la plus fraîche de l'année dans le Sud-Bénin (remontée d'eau froide, ciel couvert) — était classé dans le même panier que l'Harmattan de décembre-janvier, avec la même température de base de 33,5°C. N'importe qui connaissant le climat béninois aurait immédiatement repéré l'incohérence.

À cela s'ajoutait un second problème indépendant : la température de base changeait d'un coup au changement de mois calendaire (jusqu'à 4°C de saut sec entre le 31 et le 1er), ce qui ne correspond à aucun phénomène physique réel.

### Le choix : 4 saisons, ancrées mi-mois, interpolées

`generer_saisons_benin` distingue maintenant 4 régimes :

| Code | Saison | Période | Caractère |
|---|---|---|---|
| 0 | Grande saison des pluies | mi-mars → mi-juillet | pluies les plus intenses de l'année |
| 1 | Petite saison sèche | mi-juillet → août | la plus fraîche, ciel couvert, upwelling atlantique |
| 2 | Petite saison des pluies | septembre → octobre | pluies plus modérées que la grande saison |
| 3 | Grande saison sèche / Harmattan | novembre → mi-mars | la plus chaude et la plus poussiéreuse, pic Harmattan déc-janv |

La température de l'atelier (`generer_temperature_ambiante`) n'utilise plus un `np.where` binaire mais une **interpolation linéaire circulaire** (`_temperature_ancrage_mensuel`) entre 12 ancrages placés à mi-mois (`ConfigDonneesSynthetiques.temperature_base_mensuelle`) :

```
Jan 32.5 · Fév 34.0 · Mar 34.0 · Avr 32.0 · Mai 30.5 · Jun 29.5
Jul 28.5 · Aoû 27.5 · Sep 29.0 · Oct 30.0 · Nov 31.5 · Déc 32.0
```

**Pourquoi une interpolation par ancrages plutôt qu'une sinusoïde unique** (suggestion initiale) : le cycle réel est bimodal (deux pics, deux creux), pas une simple onde à une fréquence. Un seul `sin()` annuel aurait forcé un unique maximum et un unique minimum sur l'année, effaçant justement le phénomène qu'on cherche à représenter (le creux de la petite saison sèche). Les ancrages mensuels + interpolation permettent d'encoder la vraie forme bimodale tout en éliminant le saut en escalier — plus simple à auditer qu'un ajustement par séries de Fourier, et vérifiable point par point.

**Vérifié après implémentation** (température moyenne simulée par mois, seed 42, 30 machines, 2 ans) :

| Mois | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| °C simulé | 32.67 | 33.78 | 33.60 | 32.01 | 30.54 | 29.54 | 28.48 | **27.84** | 29.01 | 30.15 | 31.43 | 32.01 |

Août est maintenant le mois le plus frais (27.84°C) et février le plus chaud (33.78°C) — l'inversion attendue par rapport à l'ancien modèle est confirmée. La transition juin→juillet→août (29.54 → 28.48 → 27.84) est graduelle, plus de saut brutal.

### Effets de bord traités

Le passage de 2 à 4 codes de saison touchait d'autres générateurs qui utilisaient `seasons == 1` comme raccourci pour "saison sèche" :

- **Pluie** (`generer_evenements_pluie`) : probabilités de démarrage/arrêt différenciées par saison plutôt que binaire — la petite saison des pluies (2) est modélisée moins intense que la grande (0), la petite saison sèche (1) moins aride que l'Harmattan (3). Table `PROBS_TRANSITION_PLUIE`.
- **Poussière** (`generer_concentration_poussiere`) : le test `seasons == 1` (qui désignait l'ancienne "saison sèche") a été remplacé par `np.isin(seasons, [1, 3])` (les deux saisons sèches), **valeurs shape/scale inchangées** — ce commit ne touche pas encore la calibration fine de la poussière par saison, seulement sa cohérence avec le nouveau calendrier. La chaîne poussière → encrassement → surchauffe (HDF) sera traitée dans un commit dédié à la physique des pannes.
- **Bruit thermique** : air plus stable (moins de bruit) en saison sèche (1, 3) qu'en saison des pluies (0, 2) — logique conservée à l'identique, juste généralisée aux 4 codes.

`saison_benin` reste une feature catégorielle (`src/preprocessor.py` la passe déjà à un `OrdinalEncoder` + dtype `category`, aucun changement de code nécessaire côté modèle) — elle passe juste de 2 à 4 modalités.

### Ce qui n'a pas changé dans ce commit

Le mois de pic Harmattan pour la poussière (`est_harmattan = months in [12, 1]`) reste tel quel — c'est un sous-ensemble de la saison 3, cohérent avec le nouveau calendrier, pas affecté par ce fix.

---

## Réseau électrique

### Tension nominale : 230V → 220V

`ConfigDonneesSynthetiques.tension_nominale` passait de 230V (norme harmonisée européenne) à **220V**, la norme du réseau béninois (CEB/SBEE). C'est un changement cosmétique sur la valeur affichée de `niveau_tension`, **pas physique** : la déviation utilisée dans tous les calculs en aval (`stabilite_tension`, `chaleur_electrique`, l'usure liée à la tension) est `nominal − voltage`, où `voltage = nominal + charge + bruit` — la constante `nominal` s'annule algébriquement dans cette différence. Vérifié empiriquement : en régénérant le contexte exogène avec l'ancienne valeur (230V) à seed identique, `stabilite_tension` ressort à moyenne 64,18 / écart-type 19,09, contre 64,16 / 19,12 avec la nouvelle valeur — écart dans le bruit de simulation, pas un effet du changement.

### Durée des coupures : légèrement durcie

Décision actée en amont (cf. brief projet) : le réseau reste volontairement **clément** dans ce générateur — hypothèse d'une alimentation dédiée à la zone industrielle GDIZ, distincte du réseau SBEE grand public où des délestages de 10 à 15h ne sont pas rares. Un réseau aussi peu fiable que le réseau public serait incompatible avec l'exploitation industrielle continue simulée par ailleurs (activité 3×8).

Le "légèrement durci" demandé se traduit par une probabilité de persistance de la coupure relevée de 0,40 à 0,55 (`generer_indicateur_coupure`), ce qui porte la durée moyenne d'une coupure de ~50 min à **~67 min** (E[durée] = 1/(1-p) pas de 30 min). Vérifié sur une régénération complète : 77 coupures sur 2 ans, durée moyenne mesurée 63,9 min — cohérent avec la valeur théorique (l'écart vient de la taille d'échantillon, 77 événements).

Toujours très loin des 10-15h d'une coupure SBEE non dédiée — c'est voulu, et maintenant documenté plutôt qu'implicite.

---

## Physique des pannes

Quatre changements dans `EvolueurEtatMachine`, tous dans `_mettre_a_jour_physique` / `_calculer_hasards_panne` / `_gerer_phase_arret`.

### Découplage couple/vitesse

**Le problème** : `vitesse_rotation` et `couple` étaient tous les deux calculés comme une fonction directe de `niveau_activite` — corrélation mesurée ≈ +0,94. Dans un atelier textile réel, la vitesse est gouvernée par le moteur et reste quasi constante une fois la machine lancée ; c'est la **charge** (couple) qui varie avec la production, pas le régime moteur.

**Le choix** : `vitesse_rotation` ne dépend plus que de `stabilite_tension` (± bruit réduit, 25→12) — quasi constante autour de 1500 tr/min. `couple` garde sa dépendance à `niveau_activite` (c'est la charge qui doit varier avec la production, pas la vitesse).

**Vérifié** : corrélation couple/vitesse mesurée après le changement : **-0,28** (contre +0,94 avant). Vitesse : moyenne 1479 tr/min, écart-type 13 (quasi constante, comme voulu).

### Encrassement (poussière → dissipation thermique → HDF)

**Le problème** : la poussière (`concentration_poussiere`) n'avait aucun effet sur la température machine — le mode HDF (surchauffe), pourtant présenté comme le mode "Harmattan" emblématique, n'avait donc aucune chaîne causale le reliant à la saison sèche/poussiéreuse.

**Le choix** : nouvel état cumulatif `encrassement` (0-100 %, comme `usure_outil`), qui s'accumule à chaque pas actif proportionnellement à `concentration_poussiere` et à la sensibilité poussière du type de machine (`sensibilite_poussiere`, déjà utilisée pour l'usure — pas de nouveau paramètre par type inventé). L'encrassement réduit la dissipation thermique : contribue à `temperature_cible` via `facteur_chaleur_encrassement`. Remise à un résidu (10 %, pas 0 — nettoyage partiel) à chaque visite de maintenance, quelle que soit la cause de la panne réparée (une visite technicien inclut un entretien de base).

**Vérifié** : encrassement moyen plus élevé en grande saison sèche/Harmattan (38,6 %) qu'en saison des pluies (~26 %) — cohérent avec le calendrier poussière déjà en place.

### Rampe terminale de dégradation

**Le problème identifié par un test à 100 machines (mentionné dans le brief projet)** : corrélation entre temps-restant-estimé et temps-restant-réel avant panne TWF ≈ 0,03 — quasiment nulle. Cause racine identifiée en lisant le code : une fois `usure_outil` saturé à 100 % (ce qui arrive **avant** que le hasard TWF ne se déclenche, pour la majorité des machines dont le seuil personnel est élevé — distribution beta biaisée vers 80-99 %), le terme `excess` de la formule de hasard plafonne lui aussi. À partir de ce point, le hasard de panne est constant : la panne devient un pur tirage mémoryless, sans aucun signal observable pour la distinguer d'un pas au suivant.

**Le choix** : nouvel état `heures_exposition_rampe`, qui compte le temps cumulé passé dans la bande de rampe (autour du seuil personnel) — **continue de croître même quand `usure_outil` a plafonné**, plafonné lui-même à 240h (10 jours) pour éviter toute dérive non bornée. Ce compteur alimente trois choses : le hasard TWF lui-même (facteur multiplicatif qui double tous les ~2 jours d'exposition), la vibration (effet "broutement d'outil" croissant), et la température (chaleur additionnelle modeste). Le compteur n'est **pas** exposé comme colonne du dataset — seuls ses effets observables (vibration, température) le sont, pour éviter de fuiter une information que des capteurs réels n'auraient pas directement.

**Vérifié** (sur les pannes TWF du dataset régénéré, fenêtre des 240h précédant chaque panne, 38 400 points) : corrélation vibration ↔ temps-restant = **-0,21** (contre 0,03 rapporté sur l'ancien générateur). Toujours loin d'une prédiction fiable — c'est voulu, une machine industrielle n'est pas un compte à rebours déterministe — mais un signal directionnel réel et mesurable, condition nécessaire pour qu'un futur modèle de survie ait quelque chose à apprendre.

### Second processus de dégradation : usure des roulements

**Le problème** : tout le signal de dégradation passait par un seul canal (`usure_outil`) — un modèle qui apprend à repérer l'usure outil apprend, de fait, tout ce qu'il y a à apprendre. Pas réaliste : une machine textile a plusieurs modes de dégradation mécanique indépendants.

**Le choix** : nouvel état `usure_roulements` (0-100 %), piloté par le carré du ratio de vitesse (fatigue mécanique ∝ vitesse², pas la poussière ni le couple) — délibérément découplé des facteurs qui pilotent `usure_outil`. Contribue au hasard OSF (mélangé à 25 % avec l'usure outil, 75 % — `poids_roulements_osf`) et à la vibration. Remis à zéro uniquement quand OSF est la cause réparée (même logique que `usure_outil`/TWF).

**Vérifié** : corrélation `usure_outil` ↔ `usure_roulements` = **0,008** — canaux effectivement indépendants, comme voulu.

### Recalibrage — un résultat qui a d'abord été faux

Premier essai (coefficients de hasard inchangés, seuls les nouveaux canaux ajoutés) : **HDF est passé à 64,6 % des pannes** (cible 20 %), tous les autres modes écrasés. Cause : la chaleur ajoutée par l'encrassement et la rampe terminale s'additionne à la chaleur existante sans qu'aucun coefficient n'ait été réduit en face — le hasard HDF (quadratique en surchauffe soutenue) a mécaniquement explosé.

Recalibré par itération empirique (régénération complète + inspection de la répartition des causes à chaque essai, pas par calcul analytique) :

| Coefficient | Avant | Après | Raison |
|---|---|---|---|
| `coeff_risque_shd` | 0.0004 | 0.00012 | compenser la nouvelle source de chaleur (encrassement) |
| `facteur_chaleur_encrassement` | 4.5 (essai initial) | 1.8 | idem, contribution modérée plutôt que dominante |
| `coeff_risque_use` | 0.05 | 0.075 | compenser la dilution de part par HDF/OSF |
| `coeff_risque_sm` | 0.00038 | 0.0007 | compenser la dilution par le mélange avec usure_roulements |
| `coeff_risque_pe` | 0.000024 | 0.000015 | sa part croît mécaniquement quand les autres causes baissent (risques concurrents) |

Résultat final (dataset régénéré, 251 pannes / 30 machines / 2 ans, soit 8,37/machine — légèrement au-dessus de la fourchette calibrée 4-8, écart jugé acceptable) :

| Mode | Cible | Résultat |
|---|---|---|
| TWF | 40 % | 31,9 % |
| HDF | 20 % | 18,3 % |
| OSF | 15 % | 16,7 % |
| PWF | 20 % | 27,5 % |
| RNF | 5 % | 5,6 % |

Pas un alignement parfait sur les cibles (documentées dès le départ comme "cibles de calibration, pas une garantie stricte") — PWF reste au-dessus de sa cible, en partie parce qu'une fraction de ses occurrences vient d'un mécanisme séparé (choc électrique au redémarrage après coupure, `sensibilite_panne_coupure`) non affecté par ce recalibrage. Mais aucun mode n'écrase plus les autres, et la répartition reste dans le même ordre de grandeur que celle mesurée avant l'ajout de ces quatre mécanismes (TWF 34,8 / HDF 22,0 / PWF 21,6 / OSF 15,0 / RNF 6,6).

**Effet secondaire positif, non recherché** : le modèle entraîné sur ce dataset régénéré passe de F2=0,42 à F2=0,48 — cohérent avec l'ajout de signaux précurseurs réellement informatifs (chatter, encrassement) plutôt qu'un artefact de recalibrage.

---

## Maintenance préventive imparfaite

**Le problème** : les machines tournaient jusqu'à la panne, sans exception. Une fois qu'une machine entrait dans la zone à risque (usure/encrassement/roulements élevés), la seule issue possible dans les données était une panne réelle — il n'existait aucun cas où une machine à risque était interceptée avant de casser. Un modèle entraîné là-dessus apprend une relation trop propre : "indicateurs élevés ⇒ panne", sans jamais voir de contre-exemple. Réaliste ? Non : une équipe de maintenance qui fait des rondes ou des contrôles ponctuels attrape parfois un problème avant qu'il ne devienne une panne.

**Le choix** : à chaque pas, en plus du tirage de panne existant, un second tirage indépendant évalue une interception préventive — probabilité = composante routine (fixe, indépendante du risque) + composante risque (∝ `max(usure, encrassement, roulements)²`). Volontairement **imparfaite** : la probabilité reste nettement inférieure au hasard de panne réel, donc la panne l'emporte la plupart du temps — la maintenance préventive n'est pas un filet de sécurité fiable, juste une chance supplémentaire. Quand elle intercepte, le service est partiel (réduction de 50 % de l'usure/des roulements, pas une remise à neuf — `reduction_usure_preventive`) et plus court qu'une réparation de panne (médiane 2h contre 3h).

**Recalibrage** : le premier essai (`coeff_risque_preventif=0.0004`) interceptait trop de trajectoires TWF spécifiquement — la part de TWF est tombée de 32 % à 25 % des pannes, **sans réagir à une hausse du hasard TWF lui-même** (`coeff_risque_use` monté à 0,10 puis 0,12 sans effet sur la part TWF réalisée). Explication : le même signal de risque (`max(usure, ...)`) pilote à la fois le hasard TWF et l'interception préventive — booster l'un booste presque autant l'autre. Recalibré à `coeff_risque_preventif=0.0001` : la part TWF revient à un niveau proche d'avant l'ajout du mécanisme, tout en gardant un effet réel et mesurable.

**Vérifié** (50 machines, dataset régénéré) : 76 visites préventives déclenchées sur 2 ans (1,52/machine/2ans, environ une tous les 16 mois) — un rythme plausible pour un contrôle opportuniste, pas une politique de maintenance planifiée à part entière (cf. section 0 du brief projet : la politique de réentraînement/maintenance déclarée est explicitement hors périmètre). Nombre de pannes réelles : 365 sur 50 machines/2 ans (7,3/machine — dans la fourchette calibrée 4-8).

---

## Données manquantes structurées

**Le problème** : `injecter_donnees_manquantes` tirait 2 % des cellules capteurs uniformément au hasard (MCAR — *missing completely at random*), sans lien entre elles. C'est le cas le plus facile à gérer pour un pipeline de données : une valeur manquante isolée s'impute trivialement à partir de ses voisines temporelles. La réalité IoT industrielle est plus dure : un capteur qui décroche reste indisponible sur une fenêtre de temps, pas un pas sur deux au hasard.

**Le choix** : trois mécanismes indépendants, générés par (machine, capteur) — pas par cellule — pour que chaque défaillance affecte une fenêtre continue :

1. **Rafales de paquets perdus** (`rafale_nan_capteur_*`) — un capteur cesse de transmettre pendant une durée aléatoire (log-normale, médiane ~2h, jusqu'à 12h). Remplace le NaN isolé.
2. **Capteurs figés** (`capteur_fige_*`) — le capteur reste bloqué à sa dernière valeur valide pendant une durée plus longue (médiane ~6h, jusqu'à 48h) — pas de NaN, une valeur qui ne bouge plus alors que la machine, elle, continue d'évoluer.
3. **Dérive de calibration** (`derive_capteur_*`) — un biais qui croît linéairement (rampe) sur la durée de l'événement (médiane ~24h, jusqu'à 100h), signe aléatoire, amplitude proportionnelle à l'écart-type du capteur — le plus sournois des trois, aucune valeur manquante ni figée, juste une mesure silencieusement fausse.

Les trois s'appliquent dans cet ordre (dérive → figé → rafale de NaN) : un capteur qui dérive peut ensuite se figer sur sa valeur dérivée, et une perte de paquets a toujours le dernier mot (le NaN écrase tout, cohérent avec le fait qu'une valeur manquante est l'état le plus sévère).

**Calibrage** : un premier essai (`pas_moyens_duree_rafale_nan_capteur=2`, troncature par `int()`) produisait des rafales à 55 % isolées sur un seul pas — pas franchement différent du MCAR d'origine. Cause : `int()` tronque vers zéro, ce qui biaise systématiquement les durées tirées vers le bas (un tirage de 1,9 pas devient 1, pas 2). Corrigé en utilisant `np.ceil()` (cohérent avec le reste du générateur — `_gerer_phase_arret` faisait déjà ça) et en réajustant `pas_moyens_duree_rafale_nan_capteur` à 3 : la part de rafales isolées tombe à moins de 5 %.

**Vérifié** (échantillon machine 01, capteur vibration) : 83 rafales sur 2 ans, durée médiane 4 pas (2h), 4,8 % de rafales isolées à un seul pas. Sur l'ensemble du dataset (50 machines) : 0,98 % de cellules en rafale NaN, 0,47 % figées, 0,39 % en dérive — budget de corruption total (~1,8 %) proche de l'ancien flat 2 %, mais restructuré en événements réalistes plutôt qu'en bruit i.i.d.

---

## Taille du parc : 50 machines

**Le raisonnement (repris du brief projet, mesuré sur une étude antérieure, pas refait ici)** : à 30 machines, le jeu de test ne contient que ~72 pannes, avec une marge d'erreur ±11,5 points sur les métriques de recall — trop bruité pour trancher entre deux versions de modèle. À 50 machines, ~126 pannes, marge ±8,7 points. Au-delà (100 machines testé), le gain marginal (±6,2 points) ne justifie plus le coût ×2 en temps de génération et en RAM. 50 est le point d'équilibre retenu.

**Le changement** : `nb_machines` passe de 30 à 50 dans `ConfigDonneesSynthetiques`. La répartition par type (L/M/H) n'est plus un partage exact en trois (30 = 10×3), elle devient 17/17/16 — `cycle_types_machine[i % 3]` gère ça automatiquement, seul le commentaire du code a dû être corrigé pour ne plus prétendre à un partage exact.

**Effets de bord traités** : deux libellés codés en dur dans le dashboard (`ui/tabs/tab_about.py`, `dashboard.py`) affichaient "30 machines" et "1.05M" enregistrements — mis à jour vers 50 machines et 1,75M. Les mentions historiques dans ce document (sections précédentes, mesurées à l'époque sur 30 machines) sont laissées telles quelles : elles décrivent fidèlement ce qui a été mesuré au moment où c'était encore la configuration active, les réécrire en "50" serait une réécriture malhonnête de l'historique.

**Vérifié** : régénération complète en ~21s (contre ~10s à 30 machines — passage x1,67 en volume de données pour x2,1 en temps, cohérent avec le coût annoncé). 365 pannes sur 50 machines/2 ans (7,3/machine, toujours dans la fourchette 4-8), répartition par cause quasi identique à celle mesurée à 30 machines (TWF 31,0 / PWF 24,9 / HDF 19,2 / OSF 18,9 / RNF 6,0) — signe que le recalibrage des sections précédentes n'était pas surajusté au nombre de machines. Le modèle réentraîné sur ce dataset élargi n'a **pas** dépassé le champion entraîné sur 30 machines (F2 0,4729 contre 0,4904) — la gate de promotion a correctement refusé de le promouvoir. Pas un problème : les deux jeux de données ne sont pas directement comparables (composition différente), et l'évaluation événementielle propre (métrique + split validation) arrive dans la phase suivante.
