# Génération des données synthétiques — méthodologie et choix

Ce document explique **pourquoi** le générateur (`scripts/generate_synthetic_data.py`) est construit comme il l'est, et **ce qui a changé** au fil des itérations, avec la justification de chaque choix. C'est un document vivant : il est mis à jour à chaque commit qui touche la génération de données, pas rédigé après coup.

Statut des sections : les choix décrits ici sont des calibrations pour un générateur synthétique, pas des mesures de terrain. Quand un chiffre s'appuie sur une climatologie généralement documentée pour le Sud-Bénin plutôt que sur une donnée mesurée sur site GDIZ, c'est dit explicitement — pas de fausse précision.

## Sommaire

- [Calendrier saisonnier du Sud-Bénin](#calendrier-saisonnier-du-sud-bénin)
- [Réseau électrique](#réseau-électrique)

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

`generate_benin_seasons` distingue maintenant 4 régimes :

| Code | Saison | Période | Caractère |
|---|---|---|---|
| 0 | Grande saison des pluies | mi-mars → mi-juillet | pluies les plus intenses de l'année |
| 1 | Petite saison sèche | mi-juillet → août | la plus fraîche, ciel couvert, upwelling atlantique |
| 2 | Petite saison des pluies | septembre → octobre | pluies plus modérées que la grande saison |
| 3 | Grande saison sèche / Harmattan | novembre → mi-mars | la plus chaude et la plus poussiéreuse, pic Harmattan déc-janv |

La température de l'atelier (`generate_ambient_temperature`) n'utilise plus un `np.where` binaire mais une **interpolation linéaire circulaire** (`_monthly_anchor_temperature`) entre 12 ancrages placés à mi-mois (`SyntheticDataConfig.monthly_base_temp`) :

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

- **Pluie** (`generate_rain_events`) : probabilités de démarrage/arrêt différenciées par saison plutôt que binaire — la petite saison des pluies (2) est modélisée moins intense que la grande (0), la petite saison sèche (1) moins aride que l'Harmattan (3). Table `RAIN_TRANSITION_PROBS`.
- **Poussière** (`generate_dust_concentration`) : le test `seasons == 1` (qui désignait l'ancienne "saison sèche") a été remplacé par `np.isin(seasons, [1, 3])` (les deux saisons sèches), **valeurs shape/scale inchangées** — ce commit ne touche pas encore la calibration fine de la poussière par saison, seulement sa cohérence avec le nouveau calendrier. La chaîne poussière → encrassement → surchauffe (HDF) sera traitée dans un commit dédié à la physique des pannes.
- **Bruit thermique** : air plus stable (moins de bruit) en saison sèche (1, 3) qu'en saison des pluies (0, 2) — logique conservée à l'identique, juste généralisée aux 4 codes.

`benin_season` reste une feature catégorielle (`src/preprocessor.py` la passe déjà à un `OneHotEncoder`, aucun changement de code nécessaire côté modèle) — elle passe juste de 2 à 4 modalités.

### Ce qui n'a pas changé dans ce commit

Le mois de pic Harmattan pour la poussière (`is_harmattan = months in [12, 1]`) reste tel quel — c'est un sous-ensemble de la saison 3, cohérent avec le nouveau calendrier, pas affecté par ce fix.

---

## Réseau électrique

### Tension nominale : 230V → 220V

`SyntheticDataConfig.nominal_voltage` passait de 230V (norme harmonisée européenne) à **220V**, la norme du réseau béninois (CEB/SBEE). C'est un changement cosmétique sur la valeur affichée de `voltage_level`, **pas physique** : la déviation utilisée dans tous les calculs en aval (`voltage_stability`, `electrical_heat`, l'usure liée à la tension) est `nominal − voltage`, où `voltage = nominal + charge + bruit` — la constante `nominal` s'annule algébriquement dans cette différence. Vérifié empiriquement : en régénérant le contexte exogène avec l'ancienne valeur (230V) à seed identique, `voltage_stability` ressort à moyenne 64,18 / écart-type 19,09, contre 64,16 / 19,12 avec la nouvelle valeur — écart dans le bruit de simulation, pas un effet du changement.

### Durée des coupures : légèrement durcie

Décision actée en amont (cf. brief projet) : le réseau reste volontairement **clément** dans ce générateur — hypothèse d'une alimentation dédiée à la zone industrielle GDIZ, distincte du réseau SBEE grand public où des délestages de 10 à 15h ne sont pas rares. Un réseau aussi peu fiable que le réseau public serait incompatible avec l'exploitation industrielle continue simulée par ailleurs (activité 3×8).

Le "légèrement durci" demandé se traduit par une probabilité de persistance de la coupure relevée de 0,40 à 0,55 (`generate_power_loss_indicator`), ce qui porte la durée moyenne d'une coupure de ~50 min à **~67 min** (E[durée] = 1/(1-p) pas de 30 min). Vérifié sur une régénération complète : 77 coupures sur 2 ans, durée moyenne mesurée 63,9 min — cohérent avec la valeur théorique (l'écart vient de la taille d'échantillon, 77 événements).

Toujours très loin des 10-15h d'une coupure SBEE non dédiée — c'est voulu, et maintenant documenté plutôt qu'implicite.
