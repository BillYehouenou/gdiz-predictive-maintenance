"""
Module de génération de jeu_donnees synthétique pour la maintenance prédictive textile.

Architecture hybride:
- Étape A: Contexte exogène global (variables macro-environnementales)
- Étape B: Moteur d'évolution séquentiel (boucle par machine avec états N-1)
- Étape C: Agrégation en DataFrame unique et ingestion DuckDB

Framework académique: AI4I 2020 adapté à l'écosystème textile de la GDIZ.
"""

import hashlib
import logging
import pathlib
from datetime import datetime

import duckdb
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Configuration globale


class ConfigDonneesSynthetiques:
    """Configuration centralisée pour la génération de données synthétiques."""

    graine_globale = 42

    # Horizon temporel
    date_debut = datetime(2024, 1, 1)
    duree_jours = 730  # 2 ans — capture 2 cycles Harmattan + saisons des pluies
    intervalle_minutes = 30  # Pas de 30 min

    # Machines — 50 : au-delà, gain marginal sur la fiabilité des métriques de test
    # pour un coût ~x2 en temps/RAM (30 machines → marge ±11,5 pts sur le recall
    # événementiel mesuré sur le jeu de test ; 50 → ±8,7 pts ; 100 → ±6,2 pts).
    # Cf. docs/data-generation.md.
    nb_machines = 50
    ids_machines = [f"GDIZ_TEX_M_{i:02d}" for i in range(1, nb_machines + 1)]
    qualite_type_machine = ["L", "M", "H"]

    # Usure — calibré pour ~4-8 pannes / machine / 2 ans (moyenne empirique : 6.6)
    increment_usure_base = 0.005
    plage_usure_initiale = (10, 70)

    # Fragilité individuelle de chaque machine — variabilité unité-à-unité de fabrication
    forme_fragilite = 2.5

    # TWF (Tool Wear Failure) — chaque machine tire son propre seuil de rupture
    # La plupart des machines tiennent bien, certaines cassent dès 65-75 %, d'autres tiennent jusqu'à 95-99 %
    # Le hasard ne devient significatif qu'à l'approche de CE seuil personnel — pas d'un seuil universel — via une rampe lisse
    seuil_min_use = 65.0
    plage_seuil_use = 34.0  # seuils personnels dans [65, 99]
    beta_a_seuil_use = 4.0
    beta_b_seuil_use = 2.0
    bande_rampe_use = 12.0  # largeur (en points d'usure) de la rampe avant le seuil personnel
    exposant_rampe_use = 3.0
    # Recalibré (0.05 -> 0.08) pour compenser la dilution de part par les nouveaux
    # canaux HDF/OSF et par la maintenance préventive ci-dessous — cf.
    # docs/data-generation.md pour la méthode de calibration.
    coeff_risque_use = 0.08

    # HDF (Heat Dissipation Failure) — surchauffe soutenue (moyenne glissante ~4h), pas un pic instantané
    # Hasard nul sous le seuil, quadratique au-delà
    pas_fenetre_shd = 8  # 8 × 30 min = 4h
    seuil_surchauffe_shd = 38.0  # °C au-delà de l'ambiant en moyenne sur la fenêtre choisie
    # Recalibré (0.0004 -> 0.00012) : la chaîne encrassement (facteur_chaleur_encrassement)
    # ajoute une source de chaleur supplémentaire, sans baisse du coefficient HDF
    # dominait les autres causes (jusqu'à 65 % des pannes lors du premier essai).
    coeff_risque_shd = 0.00012

    # OSF (Overstrain Failure) — surcharge mécanique : couple élevé combiné à une usure déjà significative.
    # Recalibré (0.00038 -> 0.0007) pour compenser la dilution par le mélange avec
    # usure_roulements (poids_roulements_osf ci-dessous).
    coeff_risque_sm = 0.0007

    # PWF (Power Failure) — instabilité électrique routinière hors coupure franche
    # (le choc de redémarrage après blackout est géré séparément, cf. sensibilite_panne_coupure)
    # Recalibré (0.000024 -> 0.000015) : sa part croît mécaniquement quand les autres
    # causes baissent (risques concurrents), sans rapport avec un changement de PWF lui-même.
    coeff_risque_pe = 0.000015

    # RNF (Random Failure) — risque résiduel irréductible, par construction sans précurseur (miroir du RNF d'AI4I 2020)
    risque_base_pa = 0.000018

    # Dérive thermique additionnelle due au frottement d'un outil usé
    facteur_chaleur_usure = 3.0  # °C additionnels max à usure = 100 %

    # Rampe terminale de dégradation (TWF) — au-delà du seuil personnel, l'usure peut
    # plafonner (capée à 100 %) alors que la panne n'a pas encore été tirée : sans ce
    # mécanisme, le hasard redevient un pur tirage mémoryless une fois le plafond
    # atteint, non prédictible depuis les capteurs. heures_exposition_rampe compte le temps
    # cumulé passé dans la bande de rampe et fait évoluer le hasard + la vibration/
    # chaleur même quand l'usure elle-même ne bouge plus.
    heures_max_exposition_rampe_use = 240.0  # plafond (10 j) — au-delà, effet saturé
    demi_vie_heures_rampe_use = 48.0  # le hasard TWF ~double tous les 2 jours passés dans la bande
    taux_broutement_use = 0.022  # vibration additionnelle par heure d'exposition à la bande
    taux_chaleur_rampe_use = 0.014  # °C additionnels par heure d'exposition à la bande

    # Encrassement (poussière → dissipation thermique réduite → HDF)
    taux_base_encrassement = 0.06  # % d'encrassement gagné par pas, à poussiere_locale=100 et sensibilité=1
    facteur_chaleur_encrassement = 1.8  # °C additionnels max à encrassement = 100 %
    residu_reinitialisation_encrassement = 10.0  # % résiduel après une visite de maintenance (nettoyage partiel)

    # Maintenance préventive imparfaite — sans elle, les machines tournent toujours
    # jusqu'à la panne : les indicateurs précurseurs (usure, encrassement, roulements)
    # deviennent un prédicteur quasi certain de panne imminente, ce qui rend TWF
    # artificiellement facile. Une visite préventive (routine + risque, jamais
    # garantie) intercepte parfois une machine à risque avant qu'elle ne casse
    # réellement — le hasard de panne l'emporte la plupart du temps (imparfaite).
    probabilite_base_preventive = 0.00002  # composante routine, indépendante du risque
    # Recalibré (0.0004 -> 0.0001) : la première valeur interceptait trop de
    # trajectoires TWF (part TWF réduite de 32 % à 25 % des pannes, sans réaction à
    # une hausse du hasard TWF lui-même — même signal de risque pilotant les deux
    # mécanismes). Cf. docs/data-generation.md.
    coeff_risque_preventif = 0.0001  # composante risque, ∝ max(usure, encrassement, roulements)²
    heures_mediane_duree_preventive = 2.0
    heures_max_duree_preventive = 8.0
    reduction_usure_preventive = 0.5  # fraction d'usure/roulements retirée — service partiel, pas une remise à neuf

    # Second processus de dégradation indépendant de l'usure outil : usure des
    # roulements, pilotée par la vitesse/vibration plutôt que par la poussière ou le
    # couple — canal de dégradation distinct, contribue à OSF.
    taux_base_usure_roulements = 0.004
    coeff_vibration_roulements = 2.0
    poids_roulements_osf = 0.25  # part de usure_roulements_norm**3 dans le hasard OSF (reste : usure outil)

    # Parts de pannes visées par cause — cibles de calibration, pas une garantie stricte
    # (la réalisation dépend des coefficients ci-dessus, à vérifier empiriquement)
    parts_mode_panne = {"TWF": 0.40, "HDF": 0.20, "OSF": 0.15, "PWF": 0.20, "RNF": 0.05}

    # Thermique — inertie machine forte : 30 min ≈ 60 % du chemin vers T_cible
    coef_lissage_temperature = 0.55
    facteur_temperature_frottement = 25.0

    # Électrique et environnement
    # 220V = norme béninoise (réseau CEB/SBEE), pas les 230V harmonisés européens.
    tension_nominale = 220.0
    plage_variation_tension = 0.15

    # Température atelier de référence par mois (°C), ancrée mi-mois puis interpolée
    # (voir generer_temperature_ambiante) — calendrier bimodal réel du Sud-Bénin :
    # pic chaud fév-mars, creux fin juillet-août (petite saison sèche, remontée
    # d'eau froide atlantique), léger réchauffement sept-oct, Harmattan nov-mars.
    # Cf. docs/data-generation.md pour le détail et les sources.
    temperature_base_mensuelle = {
        1: 32.5,
        2: 34.0,
        3: 34.0,
        4: 32.0,
        5: 30.5,
        6: 29.5,
        7: 28.5,
        8: 27.5,
        9: 29.0,
        10: 30.0,
        11: 31.5,
        12: 32.0,
    }

    # Erreurs capteurs IoT — rafales corrélées (perte de paquets), capteurs figés,
    # dérive de calibration : remplace un MCAR uniforme (une cellule isolée sur deux
    # sans lien avec les autres), le cas le plus facile à imputer et le moins
    # réaliste pour de l'IoT industriel. Généré par (machine, capteur), pas par
    # cellule : un capteur qui décroche reste indisponible sur une fenêtre de temps
    # continue. Cf. docs/data-generation.md.
    probabilite_debut_rafale_nan_capteur = 0.0024  # probabilité de démarrage d'une rafale, par pas/capteur
    pas_moyens_duree_rafale_nan_capteur = 3.0
    sigma_duree_rafale_nan_capteur = 0.6
    plage_duree_rafale_nan_capteur = (1, 24)  # 30 min à 12h

    probabilite_debut_capteur_bloque = 0.00025
    pas_moyens_duree_capteur_bloque = 12.0
    sigma_duree_capteur_bloque = 1.0
    plage_duree_capteur_bloque = (2, 96)  # 1h à 48h

    probabilite_debut_derive_capteur = 0.00006
    pas_moyens_duree_derive_capteur = 48.0
    sigma_duree_derive_capteur = 1.0
    plage_duree_derive_capteur = (12, 200)  # 6h à 100h
    pct_max_derive_capteur = 0.5  # biais max atteint en fin de dérive, en fraction de l'écart-type du capteur

    # Chemins d'export
    dossier_sortie = pathlib.Path("data/raw")
    fichier_parquet = dossier_sortie / "gdiz_dataset.parquet"
    fichier_duckdb = dossier_sortie / "gdiz_maintenance.db"


# A - Contexte exogène global


class GenerateurContexteExogene:
    """Génère la chronologie environnementale et électrique partagée par toute l'usine."""

    def __init__(self, config: ConfigDonneesSynthetiques = None):
        self.config = config or ConfigDonneesSynthetiques()
        np.random.seed(self.config.graine_globale)
        self.nb_echantillons = self._calculer_nombre_echantillons()
        logger.info(f"Contexte exogène initialisé: {self.nb_echantillons} enregistrements à générer...")

    def _calculer_nombre_echantillons(self) -> int:
        return int(self.config.duree_jours * 24 * 60 / self.config.intervalle_minutes)

    def generer_horodatages(self) -> pd.DatetimeIndex:
        return pd.date_range(
            start=self.config.date_debut, periods=self.nb_echantillons, freq=f"{self.config.intervalle_minutes}min"
        )

    def generer_niveau_activite(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """Modélise un rythme d'activité industriel de 3 équipes en roulement 3x8."""
        heures = timestamps.hour.values
        minutes = timestamps.minute.values
        jours = timestamps.dayofweek.values

        activite_locale = np.zeros(len(timestamps))
        activite_locale = np.where((heures >= 6) & (heures < 14), 0.92, activite_locale)
        activite_locale = np.where((heures >= 14) & (heures < 22), 0.86, activite_locale)
        activite_locale = np.where((heures >= 22) | (heures < 6), 0.55, activite_locale)

        est_releve_equipe = (minutes < 20) & np.isin(heures, [6, 14, 22])
        activite_locale = np.where(est_releve_equipe, activite_locale * 0.75, activite_locale)

        bruit_operationnel = np.random.normal(0, 0.04, len(timestamps))
        activite_locale = activite_locale + bruit_operationnel
        activite_locale = np.where(jours >= 5, 0.0, activite_locale)
        return np.clip(activite_locale, 0.0, 1.0)

    def generer_saisons_benin(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """
        Calendrier bimodal réel du Sud-Bénin (climat guinéen côtier) : deux saisons des
        pluies et deux saisons sèches, pas le cycle unimodal du Bénin septentrional.
            0 = grande saison des pluies    (mi-mars → mi-juillet)
            1 = petite saison sèche         (mi-juillet → août) — la plus fraîche de l'année
            2 = petite saison des pluies    (septembre → octobre)
            3 = grande saison sèche/Harmattan (novembre → mi-mars)
        Cf. docs/data-generation.md.
        """
        mois = timestamps.month.values
        jours = timestamps.day.values
        seasons = np.full(len(timestamps), 3, dtype=int)  # défaut : grande saison sèche

        est_grande_saison_pluies = np.isin(mois, [4, 5, 6]) | ((mois == 3) & (jours >= 15)) | ((mois == 7) & (jours < 15))
        est_petite_saison_seche = ((mois == 7) & (jours >= 15)) | (mois == 8)
        est_petite_saison_pluies = np.isin(mois, [9, 10])

        seasons[est_grande_saison_pluies] = 0
        seasons[est_petite_saison_seche] = 1
        seasons[est_petite_saison_pluies] = 2
        return seasons

    def _temperature_ancrage_mensuel(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """
        Interpolation linéaire circulaire entre 12 ancrages mi-mois (config.temperature_base_mensuelle) :
        remplace l'ancien saut en escalier (jusqu'à 4°C) au changement de mois par une
        transition continue, sans supposer une forme sinusoïdale simple (le cycle annuel
        du Sud-Bénin est bimodal, pas une sinusoïde unique).
        """
        ancrages = self.config.temperature_base_mensuelle
        jour_annee_mi_mois = np.array([15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349], dtype=float)
        temperatures_ancrage = np.array([ancrages[m] for m in range(1, 13)])
        # Périodicité : on prolonge d'un point avant janvier et un après décembre
        jour_annee_etendu = np.concatenate(
            [[jour_annee_mi_mois[-1] - 365.0], jour_annee_mi_mois, [jour_annee_mi_mois[0] + 365.0]]
        )
        temperature_etendue = np.concatenate([[temperatures_ancrage[-1]], temperatures_ancrage, [temperatures_ancrage[0]]])
        jour_annee = timestamps.dayofyear.values.astype(float)
        return np.interp(jour_annee, jour_annee_etendu, temperature_etendue)

    def generer_temperature_ambiante(self, timestamps: pd.DatetimeIndex, seasons: np.ndarray) -> np.ndarray:
        """
        Température de l'atelier combinant normale saisonnière lissée et cycle jour/nuit.
        Régulation industrielle : si T_brute > 38°C, la ventilation dissipe 70 % de l'excès.
        """
        heures = timestamps.hour.values

        variation_journaliere = 4.5 * np.sin(2 * np.pi * (heures - 8) / 24)
        base_temp = self._temperature_ancrage_mensuel(timestamps)
        # Air plus stable (moins de bruit) en saison sèche (1, 3) qu'en saison des pluies (0, 2)
        est_plutot_sec = np.isin(seasons, [1, 3])
        bruit = np.where(est_plutot_sec, np.random.normal(0, 1.0, len(timestamps)), np.random.normal(0, 1.8, len(timestamps)))

        temperature_ambiante_locale = base_temp + variation_journaliere + bruit

        # Régulation/ventilation industrielle : dissipe 70 % de la chaleur excédentaire > 38°C
        temperature_ambiante_locale = np.where(
            temperature_ambiante_locale > 38.0, 38.0 + 0.30 * (temperature_ambiante_locale - 38.0), temperature_ambiante_locale
        )
        return np.clip(temperature_ambiante_locale, 23.0, 42.0)

    # (p_debut, p_arret) par saison — cf. generer_saisons_benin pour le code des 4 saisons.
    # La petite saison des pluies (2) est en moyenne moins intense que la grande (0) ;
    # la petite saison sèche (1) n'est pas aussi aride que l'Harmattan (3).
    PROBABILITES_TRANSITION_PLUIE = {
        0: (0.12, 0.40),  # grande saison des pluies
        1: (0.02, 0.65),  # petite saison sèche
        2: (0.08, 0.45),  # petite saison des pluies
        3: (0.006, 0.75),  # grande saison sèche / Harmattan
    }

    def generer_evenements_pluie(self, seasons: np.ndarray) -> np.ndarray:
        """
        Génère des épisodes de pluie cohérents via chaîne de Markov.
        Probabilités recalibrées pour un pas de 30 min (≈ 6× les valeurs 5 min).
        """
        nb_echantillons = len(seasons)
        indicateur_pluie = np.zeros(nb_echantillons, dtype=int)

        for i in range(1, nb_echantillons):
            p_debut, p_arret = self.PROBABILITES_TRANSITION_PLUIE[int(seasons[i])]

            if indicateur_pluie[i - 1] == 1:
                indicateur_pluie[i] = 1 if np.random.rand() > p_arret else 0
            else:
                indicateur_pluie[i] = 1 if np.random.rand() < p_debut else 0

        return indicateur_pluie

    def generer_humidite(self, temperature_ambiante_locale: np.ndarray, indicateur_pluie: np.ndarray) -> np.ndarray:
        humidite_base = 86 - (temperature_ambiante_locale - 30) * 2.3
        effet_pluie = indicateur_pluie * 9.5
        bruit = np.random.normal(0, 3.0, len(temperature_ambiante_locale))
        return np.clip(humidite_base + effet_pluie + bruit, 45, 98)

    def generer_concentration_poussiere(
        self, timestamps: pd.DatetimeIndex, seasons: np.ndarray, indicateur_pluie: np.ndarray
    ) -> np.ndarray:
        """Concentration de poussière : Effet Harmattan majeur en saison sèche, lavée par la pluie."""
        nb_echantillons = len(seasons)
        mois = timestamps.month.values
        est_harmattan = np.isin(mois, [12, 1])
        est_plutot_sec = np.isin(seasons, [1, 3])  # petite + grande saison sèche

        forme = np.select(condlist=[est_harmattan, est_plutot_sec], choicelist=[5.0, 2.5], default=1.5)
        echelle = np.select(condlist=[est_harmattan, est_plutot_sec], choicelist=[30.0, 12.0], default=6.0)

        poussiere_locale = np.random.gamma(forme, echelle, nb_echantillons)
        poussiere_locale = np.where(indicateur_pluie == 1, poussiere_locale * 0.15, poussiere_locale)
        return np.clip(poussiere_locale, 5, 600)

    def generer_niveau_tension(self, niveau_activite: np.ndarray) -> np.ndarray:
        tension_base = self.config.tension_nominale
        variation_charge = -15 * niveau_activite
        bruit = np.random.normal(0, 2.5, len(niveau_activite))
        return tension_base + variation_charge + bruit

    def generer_stabilite_tension(self, tension_locale: np.ndarray) -> np.ndarray:
        ecart_tension_abs = np.abs(tension_locale - self.config.tension_nominale)
        stabilite_locale = 100 * np.exp(-ecart_tension_abs / 18)
        bruit = np.random.normal(0, 1.5, len(tension_locale))
        return np.clip(stabilite_locale + bruit, 0, 100)

    def generer_indicateur_coupure(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """
        Simule des délestages SBEE et coupures de secteur.
        Probabilités recalibrées pour un pas de 30 min.
        Réseau volontairement clément (hypothèse : alimentation dédiée GDIZ, cf.
        docs/data-generation.md) mais légèrement durci par rapport à la version
        précédente : coupures en moyenne ~67 min (continuation = 0.55), toujours
        loin des 10-15h que peut atteindre une coupure SBEE non dédiée.
        """
        heures = timestamps.hour.values
        nb_echantillons = len(timestamps)

        probabilite_base = 0.0006  # ≈ 6× la valeur 5 min (0.0001)
        heures_pointe = np.isin(heures, [11, 12, 13, 14, 19, 20, 21, 22])
        probabilite_panne = np.where(heures_pointe, probabilite_base * 8, probabilite_base)

        coupure = (np.random.random(nb_echantillons) < probabilite_panne).astype(int)

        # E[durée] = 1/(1-p) pas ≈ 2.2 pas ≈ 67 min (contre ~50 min avant durcissement)
        for i in range(1, nb_echantillons):
            if coupure[i - 1] == 1 and np.random.random() < 0.55:
                coupure[i] = 1

        return coupure

    def generer_contexte_global(self) -> pd.DataFrame:
        timestamps = self.generer_horodatages()
        seasons = self.generer_saisons_benin(timestamps)
        indicateur_pluie = self.generer_evenements_pluie(seasons)
        temperature_ambiante_locale = self.generer_temperature_ambiante(timestamps, seasons)
        humidite = self.generer_humidite(temperature_ambiante_locale, indicateur_pluie)
        poussiere_locale = self.generer_concentration_poussiere(timestamps, seasons, indicateur_pluie)
        niveau_activite = self.generer_niveau_activite(timestamps)
        tension_locale = self.generer_niveau_tension(niveau_activite)
        stabilite_tension = self.generer_stabilite_tension(tension_locale)
        coupure = self.generer_indicateur_coupure(timestamps)

        df_global = pd.DataFrame(
            {
                "timestamp": timestamps,
                "niveau_activite": niveau_activite,
                "saison_benin": seasons,
                "temperature_ambiante": temperature_ambiante_locale,
                "indicateur_pluie": indicateur_pluie,
                "humidite": humidite,
                "concentration_poussiere": poussiere_locale,
                "niveau_tension": tension_locale,
                "stabilite_tension": stabilite_tension,
                "indicateur_coupure": coupure,
            }
        )
        logger.info(f"Contexte exogène généré avec succès ({df_global.shape[0]} enregistrements)")
        return df_global


# B - Moteur d'évolution chronologique par machine


class EvolueurEtatMachine:
    """Gère l'évolution d'état chronologique d'une machine (Dépendance physique N-1)."""

    def __init__(
        self,
        id_machine: str,
        type_machine: str,
        df_global: pd.DataFrame,
        config: ConfigDonneesSynthetiques = None,
        parametres_surcharge: dict = None,
    ):
        self.id_machine = id_machine
        self.type_machine = type_machine
        self.df_global = df_global
        self.config = config or ConfigDonneesSynthetiques()

        # RNG isolé par machine — évite toute pollution croisée des flux aléatoires
        # Hash déterministe (md5) plutôt que hash() builtin : hash() est salé
        # aléatoirement par process (PYTHONHASHSEED) sauf s'il est fixé explicitement,
        # ce qui casserait la reproductibilité entre deux exécutions.
        hash_machine = int(hashlib.md5(id_machine.encode("utf-8")).hexdigest(), 16) % 10000
        graine_machine = self.config.graine_globale + hash_machine
        self.rng = np.random.default_rng(graine_machine)

        # Fragilité individuelle (fragilite model) — tirée une fois par machine,
        # n'affecte que le hasard de panne TWF, pas la vitesse d'usure elle-même.
        forme = self.config.forme_fragilite
        self.fragilite = float(self.rng.gamma(forme, 1.0 / forme))

        # Seuil de rupture personnel (TWF) — variabilité unité-à-unité de fabrication :
        # certaines machines cassent dès 65-75 % d'usure, d'autres tiennent jusqu'à 95-99 %.
        self.seuil_use = self.config.seuil_min_use + self.config.plage_seuil_use * self.rng.beta(
            self.config.beta_a_seuil_use, self.config.beta_b_seuil_use
        )

        self.nb_echantillons = len(df_global)
        self.tableau_temperature_ambiante = df_global["temperature_ambiante"].values
        self.tableau_poussiere = df_global["concentration_poussiere"].values
        self.tableau_stabilite = df_global["stabilite_tension"].values
        self.tableau_tension = df_global["niveau_tension"].values
        self.tableau_niveau_activite = df_global["niveau_activite"].values
        self.tableau_coupures = df_global["indicateur_coupure"].values
        self._definir_parametres_machine()
        if parametres_surcharge:
            self.params.update(parametres_surcharge)
        self._initialiser_tableaux_etat()

    def _definir_parametres_machine(self):
        parametres_type = {
            "L": {  # Entrée de gamme / Rustique — aucune protection électronique
                "multiplicateur_usure": 1.3,
                "decalage_temperature_cible": 3.0,
                "bruit_vibration": 0.25,
                "sensibilite_poussiere": 1.5,
                "sensibilite_tension": 0.5,
                "malus_redemarrage_coupure": 0.8,
                # Coupure réseau : le choc mécanique au redémarrage est violent pour L
                "sensibilite_panne_coupure": 1.0,
            },
            "M": {  # Milieu de gamme / Standard
                "multiplicateur_usure": 1.0,
                "decalage_temperature_cible": 0.0,
                "bruit_vibration": 0.15,
                "sensibilite_poussiere": 1.0,
                "sensibilite_tension": 1.0,
                "malus_redemarrage_coupure": 0.4,
                "sensibilite_panne_coupure": 0.8,
            },
            "H": {  # Haut de gamme / Précision Électronique + Onduleur UPS industriel
                "multiplicateur_usure": 0.7,
                "decalage_temperature_cible": -2.0,
                "bruit_vibration": 0.08,
                "sensibilite_poussiere": 0.4,
                # sensibilite_tension conservé pour la physique (usure, chaleur) :
                # l'UPS ne supprime pas les effets physiques, il protège des pannes.
                "sensibilite_tension": 2.0,
                "malus_redemarrage_coupure": 0.2,
                # Coupure réseau : l'UPS amortit la surtension → panne sur blackout rare
                "sensibilite_panne_coupure": 0.2,
                # Instabilité routinière : l'onduleur absorbe les creux < blackout total
                # sensibilité effective aux pannes électriques spontanées quasi nulle
                "sensibilite_electrique_onduleur": 0.1,
            },
        }
        self.params = parametres_type[self.type_machine]

    def _initialiser_tableaux_etat(self):
        self.usure_outil = np.zeros(self.nb_echantillons)
        self.temperature_process = np.zeros(self.nb_echantillons)
        self.vitesse_rotation = np.zeros(self.nb_echantillons)
        self.couple = np.zeros(self.nb_echantillons)
        self.vibration = np.zeros(self.nb_echantillons)
        self.failure = np.zeros(self.nb_echantillons, dtype=int)
        self.mode_panne = np.full(self.nb_echantillons, "", dtype=object)

        # Encrassement (point 7) et usure des roulements (point 11) : mêmes limites
        # 0-100 % que usure_outil, mais accumulation et remise à neuf indépendantes.
        self.encrassement = np.zeros(self.nb_echantillons)
        self.usure_roulements = np.zeros(self.nb_echantillons)
        # Temps cumulé (h) passé dans la bande de rampe TWF (point 8) — remis à zéro
        # à chaque visite de maintenance, quelle que soit la cause de la panne.
        self.heures_exposition_rampe = np.zeros(self.nb_echantillons)

        self.usure_outil[0] = self.rng.uniform(*self.config.plage_usure_initiale)
        self.temperature_process[0] = self.tableau_temperature_ambiante[0] + 5.0

    def _mettre_a_jour_physique(self, idx: int, redemarre_apres_coupure: bool = False):
        """Calcule les indicateurs physiques de la machine lorsqu'elle est en fonctionnement."""
        indice_activite = self.tableau_niveau_activite[idx]
        pourcentage_stabilite = self.tableau_stabilite[idx] / 100.0
        ecart_tension = max(0, self.config.tension_nominale - self.tableau_tension[idx])

        # 1. Vitesse & Couple — découplés : dans un atelier textile réel, la vitesse
        # est gouvernée par le moteur et reste quasi constante une fois la machine
        # lancée ; c'est la CHARGE (couple) qui varie avec l'activité/production, pas
        # le régime moteur. (Avant : les deux étaient pilotés par niveau_activite,
        # corrélation mesurée +0.94 — cf. docs/data-generation.md.)
        vitesse_base = 1500.0
        self.vitesse_rotation[idx] = vitesse_base * (0.97 + 0.03 * pourcentage_stabilite)
        self.vitesse_rotation[idx] += self.rng.normal(0, 12)
        self.vitesse_rotation[idx] = max(0, self.vitesse_rotation[idx])

        couple_base = 45.0
        self.couple[idx] = couple_base * indice_activite * (1.3 - 0.3 * pourcentage_stabilite)
        self.couple[idx] += self.rng.normal(0, 3)
        self.couple[idx] = max(0, self.couple[idx])

        # 2. Usure cumulative (outil)
        usure_precedente = self.usure_outil[idx - 1]
        if redemarre_apres_coupure:
            usure_precedente = min(usure_precedente + self.params["malus_redemarrage_coupure"], 100)

        dust_factor = 1.0 + 0.004 * self.tableau_poussiere[idx] * self.params["sensibilite_poussiere"]
        voltage_factor = 1.0 + 0.6 * (1.0 - pourcentage_stabilite) * self.params["sensibilite_tension"]
        increment_usure = self.config.increment_usure_base * self.params["multiplicateur_usure"]
        increment_usure *= dust_factor * voltage_factor
        self.usure_outil[idx] = min(usure_precedente + increment_usure, 100)

        # 2b. Encrassement — accumulation indépendante liée à l'exposition à la
        # poussière, remise à un résidu (residu_reinitialisation_encrassement) à chaque visite de
        # maintenance (cf. _gerer_phase_arret). Alimente la chaleur, pas l'usure
        # outil : deux canaux physiques distincts vers HDF vs TWF.
        increment_encrassement = (
            self.config.taux_base_encrassement * (self.tableau_poussiere[idx] / 100.0) * self.params["sensibilite_poussiere"]
        )
        self.encrassement[idx] = min(self.encrassement[idx - 1] + increment_encrassement, 100.0)

        # 2c. Usure des roulements — second processus de dégradation, indépendant de
        # l'usure outil : piloté par la vitesse (fatigue mécanique), pas la poussière
        # ni le couple. Remis à zéro seulement quand OSF est la cause réparée
        # (cf. _gerer_phase_arret).
        ratio_vitesse = self.vitesse_rotation[idx] / vitesse_base
        increment_usure_roulements = (
            self.config.taux_base_usure_roulements * (ratio_vitesse**2) * self.params["multiplicateur_usure"]
        )
        self.usure_roulements[idx] = min(self.usure_roulements[idx - 1] + increment_usure_roulements, 100.0)

        # 2d. Rampe terminale — temps cumulé passé dans la bande de rampe TWF (autour
        # du seuil personnel). Continue de s'accumuler même quand usure_outil plafonne à
        # 100 % : sans ça, le hasard de panne redevient un pur tirage mémoryless une
        # fois l'usure saturée, invisible depuis les capteurs (cf. point 8 — condition
        # nécessaire pour tout modèle de survie futur).
        dans_bande_rampe = self.usure_outil[idx] >= (self.seuil_use - self.config.bande_rampe_use)
        if dans_bande_rampe:
            step_hours = self.config.intervalle_minutes / 60.0
            self.heures_exposition_rampe[idx] = min(
                self.heures_exposition_rampe[idx - 1] + step_hours, self.config.heures_max_exposition_rampe_use
            )
        else:
            self.heures_exposition_rampe[idx] = 0.0

        # 3. Température Machine
        temperature_precedente = self.temperature_process[idx - 1]
        chaleur_electrique = (ecart_tension * 0.15) * self.params["sensibilite_tension"]
        # Frottement accru par l'usure : dérive thermique supplémentaire, corrélée à
        # la vibration — donne deux indicateurs précurseurs conjoints pour TWF.
        chaleur_usure = self.config.facteur_chaleur_usure * (self.usure_outil[idx] / 100.0)
        # Encrassement : dissipation thermique réduite par la poussière accumulée.
        chaleur_encrassement = self.config.facteur_chaleur_encrassement * (self.encrassement[idx] / 100.0)
        # Rampe terminale : chaleur additionnelle qui continue de croître avec le
        # temps passé en zone critique, même quand usure_outil a plafonné.
        chaleur_rampe = self.config.taux_chaleur_rampe_use * self.heures_exposition_rampe[idx]
        temperature_cible = (
            self.tableau_temperature_ambiante[idx]
            + (self.config.facteur_temperature_frottement * (self.couple[idx] / 35.0))
            + self.params["decalage_temperature_cible"]
            + chaleur_electrique
            + chaleur_usure
            + chaleur_encrassement
            + chaleur_rampe
        )
        alpha = self.config.coef_lissage_temperature
        self.temperature_process[idx] = alpha * temperature_cible + (1 - alpha) * temperature_precedente

        # 4. Vibrations
        usure_normalisee = self.usure_outil[idx] / 100.0
        usure_roulements_normalisee = self.usure_roulements[idx] / 100.0
        vibration_base = 0.4 * (self.vitesse_rotation[idx] / vitesse_base) + 0.4 * (1.0 - pourcentage_stabilite)
        if redemarre_apres_coupure:
            vibration_base += self.rng.uniform(3.0, 5.0) * self.params["multiplicateur_usure"]
        effet_usure = 4.5 * (usure_normalisee**2)
        effet_roulements = self.config.coeff_vibration_roulements * (usure_roulements_normalisee**2)
        # Broutement d'outil (chatter) : s'amplifie avec le temps passé en zone
        # critique — signal observable qui continue de monter même quand effet_usure
        # a plafonné (point 8).
        effet_broutement = self.config.taux_broutement_use * self.heures_exposition_rampe[idx]
        bruit = self.rng.normal(0, self.params["bruit_vibration"])
        self.vibration[idx] = np.clip(vibration_base + effet_usure + effet_roulements + effet_broutement + bruit, 0.05, 15.0)

    def _calculer_risques_panne(self, idx: int) -> dict:
        """
        Évalue le hasard de panne par cause au pas actuel (modèle à risques concurrents,
        façon AI4I 2020 : TWF, HDF, OSF, PWF, RNF). Chaque cause a sa propre signature
        précurseur ; aucune ne repose sur un seuil dur déterministe.
        """
        wear = self.usure_outil[idx]
        usure_normalisee = wear / 100.0
        couple = self.couple[idx]
        pourcentage_stabilite = self.tableau_stabilite[idx] / 100.0

        # TWF — rampe lisse autour du seuil de rupture personnel de la machine
        # (self.seuil_use, tiré une fois à l'initialisation) : pas de seuil
        # universel, étalement réel du moment de panne entre machines fragiles et
        # robustes. La fragilité (self.fragilite) module l'intensité de la rampe.
        band = self.config.bande_rampe_use
        excess = max(0.0, (wear - (self.seuil_use - band)) / band)
        # Facteur temporel : le hasard continue de croître avec le temps cumulé passé
        # dans la bande de rampe, même quand `excess` a plafonné parce que l'usure est
        # saturée à 100 % — sans ça, une fois saturée, la panne redevient un pur
        # tirage mémoryless (cf. heures_exposition_rampe dans _mettre_a_jour_physique, point 8).
        facteur_temps = 1.0 + self.heures_exposition_rampe[idx] / self.config.demi_vie_heures_rampe_use
        twf = (
            self.config.coeff_risque_use
            * (excess**self.config.exposant_rampe_use)
            * facteur_temps
            * self.fragilite
            * self.params["multiplicateur_usure"]
        )

        # HDF — surchauffe soutenue sur une fenêtre glissante de ~4h, pas un pic
        # instantané : la dissipation thermique insuffisante est un phénomène lent.
        window = min(idx, self.config.pas_fenetre_shd)
        overheat = self.temperature_process[idx - window : idx + 1] - self.tableau_temperature_ambiante[idx - window : idx + 1]
        surchauffe_soutenue = max(0.0, float(overheat.mean()) - self.config.seuil_surchauffe_shd)
        hdf = self.config.coeff_risque_shd * surchauffe_soutenue**2

        # OSF — surcharge mécanique : couple élevé combiné à une usure déjà
        # significative. usure_normalisee**3 supprime naturellement le hasard à usure
        # faible, sans seuil dur explicite. Combine usure outil et usure des
        # roulements (second processus de dégradation indépendant, point 11) — pas
        # seulement l'usure outil, sinon OSF ne fait que dupliquer le signal TWF.
        usure_roulements_normalisee = self.usure_roulements[idx] / 100.0
        terme_usure_osf = (
            1.0 - self.config.poids_roulements_osf
        ) * usure_normalisee**3 + self.config.poids_roulements_osf * usure_roulements_normalisee**3
        osf = self.config.coeff_risque_sm * (couple / 45.0) * terme_usure_osf

        # PWF — instabilité électrique routinière hors coupure franche (le choc
        # de redémarrage après blackout est géré séparément dans evoluer_etats).
        sensibilite_electrique = self.params.get("sensibilite_electrique_onduleur", self.params["sensibilite_tension"])
        pwf = self.config.coeff_risque_pe * max(0.0, 1.0 - pourcentage_stabilite) * sensibilite_electrique

        # RNF — risque résiduel irréductible, par construction sans précurseur.
        rnf = self.config.risque_base_pa

        return {"TWF": twf, "HDF": hdf, "OSF": osf, "PWF": pwf, "RNF": rnf}

    def _gerer_phase_arret(self, index_panne: int) -> int:
        """
        Met la machine en sommeil pour maintenance suite à une panne.
        Durée générée via loi Log-Normale : médiane ≈ 3h, queue jusqu'à 48h.
        Refroidissement passif accéléré : alpha = 0.70 (retour rapide à l'ambiant en 30 min).
        """
        alpha_refroidissement = 0.70

        # Log-Normale : médiane = exp(ln(3)) = 3h, σ=1.2 → P99 ≈ 48h naturellement
        duree_heures = float(self.rng.lognormal(mean=np.log(3.0), sigma=1.2))
        duree_heures = float(np.clip(duree_heures, 0.5, 48.0))
        nb_pas = int(np.ceil(duree_heures * 60 / self.config.intervalle_minutes))

        debut_arret = index_panne + 1
        fin_arret = min(debut_arret + nb_pas, self.nb_echantillons)

        for idx in range(debut_arret, fin_arret):
            self.vitesse_rotation[idx] = 0.0
            self.couple[idx] = 0.0
            self.vibration[idx] = np.abs(self.rng.normal(0, 0.01))
            self.failure[idx] = 0
            self.usure_outil[idx] = self.usure_outil[idx - 1]
            self.encrassement[idx] = self.encrassement[idx - 1]
            self.usure_roulements[idx] = self.usure_roulements[idx - 1]
            self.heures_exposition_rampe[idx] = self.heures_exposition_rampe[idx - 1]

            # Refroidissement passif accéléré : en 30 min, la machine retombe quasi à l'ambiant
            self.temperature_process[idx] = (
                alpha_refroidissement * self.tableau_temperature_ambiante[idx]
                + (1 - alpha_refroidissement) * self.temperature_process[idx - 1]
            )

        if fin_arret < self.nb_echantillons:
            cause = self.mode_panne[index_panne]
            # Remise à neuf de l'outil — seulement si la panne réparée est l'outil
            # lui-même (TWF). Les autres causes réparent un autre composant : l'usure
            # de l'outil continue de s'accumuler sans être remise à zéro, sinon les
            # pannes non-mécaniques interrompraient indéfiniment la montée en usure et
            # TWF ne se déclencherait jamais.
            if cause == "TWF":
                self.usure_outil[fin_arret - 1] = 0.0
            # Remise à neuf des roulements — seulement si OSF est la cause réparée
            # (même logique que TWF/usure_outil, second processus indépendant).
            if cause == "OSF":
                self.usure_roulements[fin_arret - 1] = 0.0
            # Toute visite de maintenance inclut un entretien de base, quelle que soit
            # la cause : nettoyage partiel (encrassement ramené au plus à un résidu,
            # jamais remonté artificiellement) et contrôle de sécurité (zone de rampe
            # réinitialisée).
            self.encrassement[fin_arret - 1] = min(
                self.encrassement[fin_arret - 1], self.config.residu_reinitialisation_encrassement
            )
            self.heures_exposition_rampe[fin_arret - 1] = 0.0

        return fin_arret

    def _probabilite_maintenance_preventive(self, idx: int) -> float:
        """
        Probabilité qu'une visite de maintenance préventive intercepte la machine à
        ce pas — composante routine (calendaire, indépendante du risque) + composante
        risque (usure/encrassement/roulements). Reste délibérément faible devant le
        hasard de panne réel : l'interception est possible, pas garantie (imparfaite).
        """
        score_risque = max(self.usure_outil[idx], self.encrassement[idx], self.usure_roulements[idx]) / 100.0
        return self.config.probabilite_base_preventive + self.config.coeff_risque_preventif * score_risque**2

    def _gerer_phase_maintenance_preventive(self, event_idx: int) -> int:
        """
        Visite de maintenance préventive : pas une panne (failure reste à 0, aucun
        mode_panne enregistré), plus courte qu'une réparation, et surtout
        imparfaite — réduit l'usure/les roulements sans les remettre à neuf,
        contrairement à une réparation de panne qui répare le composant en cause.
        """
        duree_heures = float(self.rng.lognormal(mean=np.log(self.config.heures_mediane_duree_preventive), sigma=0.6))
        duree_heures = float(np.clip(duree_heures, 0.5, self.config.heures_max_duree_preventive))
        nb_pas = int(np.ceil(duree_heures * 60 / self.config.intervalle_minutes))

        start = event_idx + 1
        end = min(start + nb_pas, self.nb_echantillons)
        alpha_refroidissement = 0.70

        for idx in range(start, end):
            self.vitesse_rotation[idx] = 0.0
            self.couple[idx] = 0.0
            self.vibration[idx] = np.abs(self.rng.normal(0, 0.01))
            self.failure[idx] = 0
            self.usure_outil[idx] = self.usure_outil[idx - 1]
            self.encrassement[idx] = self.encrassement[idx - 1]
            self.usure_roulements[idx] = self.usure_roulements[idx - 1]
            self.heures_exposition_rampe[idx] = self.heures_exposition_rampe[idx - 1]
            self.temperature_process[idx] = (
                alpha_refroidissement * self.tableau_temperature_ambiante[idx]
                + (1 - alpha_refroidissement) * self.temperature_process[idx - 1]
            )

        if end < self.nb_echantillons:
            # Service partiel — réduit sans remettre à neuf (contrairement à une
            # réparation de panne, cf. _gerer_phase_arret).
            self.usure_outil[end - 1] *= 1.0 - self.config.reduction_usure_preventive
            self.usure_roulements[end - 1] *= 1.0 - self.config.reduction_usure_preventive
            self.encrassement[end - 1] = min(self.encrassement[end - 1], self.config.residu_reinitialisation_encrassement)
            self.heures_exposition_rampe[end - 1] = 0.0

        return end

    def evoluer_etats(self) -> pd.DataFrame:
        """Exécute la boucle d'évolution séquentielle sur la chronologie de la machine."""
        index_fin_maintenance = 0

        for idx in range(1, self.nb_echantillons):
            if idx < index_fin_maintenance:
                continue

            # Interception des délestages et coupures de courant
            if self.tableau_coupures[idx] == 1:
                self.vitesse_rotation[idx] = 0.0
                self.couple[idx] = 0.0
                self.vibration[idx] = np.abs(self.rng.normal(0, 0.01))
                self.usure_outil[idx] = self.usure_outil[idx - 1]
                self.encrassement[idx] = self.encrassement[idx - 1]
                self.usure_roulements[idx] = self.usure_roulements[idx - 1]
                self.heures_exposition_rampe[idx] = self.heures_exposition_rampe[idx - 1]
                self.failure[idx] = 0

                alpha = self.config.coef_lissage_temperature
                self.temperature_process[idx] = (
                    alpha * self.tableau_temperature_ambiante[idx] + (1 - alpha) * self.temperature_process[idx - 1]
                )

                if self.tableau_coupures[idx - 1] == 0:
                    # L=4%, M=3.2%, H=0.8% — H protégé par l'UPS au choc de coupure
                    probabilite_choc_electrique = 0.04 * self.params["sensibilite_panne_coupure"]
                    if self.rng.random() < probabilite_choc_electrique:
                        self.failure[idx] = 1
                        self.mode_panne[idx] = "PWF"
                        index_fin_maintenance = self._gerer_phase_arret(idx)
                continue

            redemarre_apres_coupure = self.tableau_coupures[idx - 1] == 1

            # Arrêts planifiés (week-end, activité nulle)
            if self.tableau_niveau_activite[idx] == 0.0:
                self.vitesse_rotation[idx] = 0.0
                self.couple[idx] = 0.0
                self.vibration[idx] = np.abs(self.rng.normal(0, 0.01))
                self.usure_outil[idx] = self.usure_outil[idx - 1]
                self.encrassement[idx] = self.encrassement[idx - 1]
                self.usure_roulements[idx] = self.usure_roulements[idx - 1]
                self.heures_exposition_rampe[idx] = self.heures_exposition_rampe[idx - 1]
                self.failure[idx] = 0

                alpha = self.config.coef_lissage_temperature
                self.temperature_process[idx] = (
                    alpha * self.tableau_temperature_ambiante[idx] + (1 - alpha) * self.temperature_process[idx - 1]
                )
                continue

            # Régime de fonctionnement normal
            self._mettre_a_jour_physique(idx, redemarre_apres_coupure=redemarre_apres_coupure)

            hazards = self._calculer_risques_panne(idx)
            total_hazard = sum(hazards.values())
            probabilite_panne = min(total_hazard, 1.0)
            if self.rng.random() < probabilite_panne:
                self.failure[idx] = 1
                modes = list(hazards.keys())
                weights = np.array(list(hazards.values()))
                mode_probs = weights / total_hazard if total_hazard > 0 else np.full(len(modes), 1.0 / len(modes))
                self.mode_panne[idx] = self.rng.choice(modes, p=mode_probs)
                index_fin_maintenance = self._gerer_phase_arret(idx)
            elif self.rng.random() < self._probabilite_maintenance_preventive(idx):
                # Maintenance préventive imparfaite : intercepte parfois une machine à
                # risque avant la panne réelle — cf. _probabilite_maintenance_preventive.
                index_fin_maintenance = self._gerer_phase_maintenance_preventive(idx)

        return pd.DataFrame(
            {
                "id_machine": self.id_machine,
                "type_machine": self.type_machine,
                "timestamp": self.df_global["timestamp"],
                "vitesse_rotation": self.vitesse_rotation,
                "couple": self.couple,
                "usure_outil": self.usure_outil,
                "encrassement": self.encrassement,
                "usure_roulements": self.usure_roulements,
                "temperature_process": self.temperature_process,
                "vibration": self.vibration,
                "panne": self.failure,
                "mode_panne": self.mode_panne,
            }
        )


# C - Agrégation, masquage IoT et persistence des données


class AgregateurDonnees:
    """Combine les historiques des machines et orchestre la persistence des données."""

    def __init__(self, config: ConfigDonneesSynthetiques = None):
        self.config = config or ConfigDonneesSynthetiques()

    def _appliquer_rafales(
        self,
        values: np.ndarray,
        start_prob: float,
        duration_mean_steps: float,
        duration_sigma: float,
        duration_range: tuple,
        mode: str,
    ) -> tuple[np.ndarray, int]:
        """
        Applique des rafales indépendantes à une série 1D d'un (machine, capteur) :
        mode="nan" (perte de paquets) ou mode="stuck" (capteur figé à sa valeur de
        début de rafale). Durée tirée d'une loi log-normale, comme les autres durées
        d'événements du générateur (coupures, maintenance).
        """
        n = len(values)
        out = values.copy()
        starts = np.flatnonzero(np.random.random(n) < start_prob)
        dmin, dmax = duration_range
        affected = 0
        for s in starts:
            duration = int(np.ceil(np.clip(np.random.lognormal(np.log(duration_mean_steps), duration_sigma), dmin, dmax)))
            end = min(s + duration, n)
            if mode == "nan":
                out[s:end] = np.nan
            else:  # "stuck" — valeur d'origine (pré-corruption) au début de la rafale
                out[s:end] = values[s]
            affected += end - s
        return out, affected

    def _appliquer_derive(self, values: np.ndarray, echelle: float) -> tuple[np.ndarray, int]:
        """
        Dérive de calibration : biais qui croît linéairement (rampe 0 → max) sur la
        durée de l'événement, signe aléatoire, amplitude proportionnelle à l'écart-type
        du capteur — pas de valeur manquante, un biais silencieux, plus difficile à
        repérer qu'un NaN.
        """
        n = len(values)
        out = values.copy()
        starts = np.flatnonzero(np.random.random(n) < self.config.probabilite_debut_derive_capteur)
        dmin, dmax = self.config.plage_duree_derive_capteur
        affected = 0
        for s in starts:
            duration = int(
                np.ceil(
                    np.clip(
                        np.random.lognormal(
                            np.log(self.config.pas_moyens_duree_derive_capteur), self.config.sigma_duree_derive_capteur
                        ),
                        dmin,
                        dmax,
                    )
                )
            )
            end = min(s + duration, n)
            sign = 1.0 if np.random.random() < 0.5 else -1.0
            ramp = np.linspace(0.0, 1.0, end - s)
            out[s:end] += sign * echelle * self.config.pct_max_derive_capteur * ramp
            affected += end - s
        return out, affected

    def injecter_donnees_manquantes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Simule des défaillances de capteurs IoT réalistes plutôt qu'un MCAR uniforme
        (une cellule isolée sur deux sans lien avec les autres, le cas le plus facile
        à imputer) : rafales de paquets perdus, capteurs figés, dérive de calibration.
        Généré indépendamment par (machine, capteur) — pas par cellule.
        """
        df_copy = df.copy()
        colonnes_capteurs = [
            "usure_outil",
            "temperature_process",
            "vitesse_rotation",
            "couple",
            "vibration",
            "temperature_ambiante",
            "humidite",
            "concentration_poussiere",
        ]
        colonnes_capteurs = [col for col in colonnes_capteurs if col in df_copy.columns]

        if not colonnes_capteurs or "id_machine" not in df_copy.columns or df_copy.empty:
            return df_copy

        # Extraction explicite en float → corruption → réassignation propre. copy=True :
        # sous pandas 3.x (copy-on-write par défaut), to_numpy() sans copie explicite
        # peut renvoyer un tableau read-only et faire échouer l'affectation ci-dessous.
        donnees_capteurs = df_copy[colonnes_capteurs].to_numpy(dtype=float, copy=True)

        n_nan = n_stuck = n_drift = 0
        for positions in df_copy.groupby("id_machine", sort=False).indices.values():
            positions = np.asarray(positions)
            for c in range(len(colonnes_capteurs)):
                values = donnees_capteurs[positions, c]
                echelle = float(np.nanstd(values)) or 1.0

                values, affected = self._appliquer_derive(values, echelle)
                n_drift += affected

                values, affected = self._appliquer_rafales(
                    values,
                    self.config.probabilite_debut_capteur_bloque,
                    self.config.pas_moyens_duree_capteur_bloque,
                    self.config.sigma_duree_capteur_bloque,
                    self.config.plage_duree_capteur_bloque,
                    mode="stuck",
                )
                n_stuck += affected

                values, affected = self._appliquer_rafales(
                    values,
                    self.config.probabilite_debut_rafale_nan_capteur,
                    self.config.pas_moyens_duree_rafale_nan_capteur,
                    self.config.sigma_duree_rafale_nan_capteur,
                    self.config.plage_duree_rafale_nan_capteur,
                    mode="nan",
                )
                n_nan += affected

                donnees_capteurs[positions, c] = values

        df_copy[colonnes_capteurs] = donnees_capteurs
        logger.info(
            f"Défaillance capteurs : {n_nan} valeurs manquantes (rafales), {n_stuck} figées, {n_drift} en dérive de calibration."
        )
        return df_copy

    def sauvegarder_parquet(self, df: pd.DataFrame):
        self.config.dossier_sortie.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.config.fichier_parquet, compression="snappy", index=False)
        size_mb = self.config.fichier_parquet.stat().st_size / (1024**2)
        logger.info(f"Fichier {self.config.fichier_parquet} sauvegardé ({size_mb:.2f} MB)")

    def inserer_duckdb(self, df: pd.DataFrame):
        db_path = pathlib.Path(self.config.fichier_duckdb).resolve()
        try:
            db_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Impossible de créer le dossier pour la BDD : {db_path.parent}. Erreur : {e}")
            raise IOError(f"Échec de l'arborescence système : {e}")

        try:
            logger.info(f"Connexion à la base DuckDB : {db_path}")
            conn = duckdb.connect(str(db_path))
            try:
                table_name = "factory_telemetry"
                conn.execute(f"DROP TABLE IF EXISTS {table_name}")
                conn.register("df_arrow", df)
                conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df_arrow")
                conn.execute(f"CREATE INDEX idx_machine_time ON {table_name}(id_machine, timestamp)")
                rows = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchall()[0][0]
                logger.info(f"Base de données DuckDB mise à jour ({rows:,} lignes).")
            finally:
                conn.close()
        except duckdb.Error as sql_error:
            logger.critical(f"Erreur SQL DuckDB : {sql_error}")
            raise sql_error
        except Exception as general_error:
            logger.critical(f"Erreur système inattendue lors de l'ingestion DuckDB : {general_error}")
            raise general_error


# Orchestrateur principal


class OrchestrateurDatasetSynthetique:
    """Pilote suprême du pipeline d'ingénierie des données synthétiques."""

    def __init__(self, graine_aleatoire: int | None = None):
        self.config = ConfigDonneesSynthetiques()
        if graine_aleatoire is not None:
            self.config.graine_globale = graine_aleatoire
        # Pas de np.random.seed() ici : GenerateurContexteExogene.__init__ reseed
        # déjà avec self.config.graine_globale juste après — l'appel ici était redondant
        # et source de confusion sur la source de vérité du seed.

    def generer(self) -> pd.DataFrame:
        generateur_exogene = GenerateurContexteExogene(self.config)
        df_global = generateur_exogene.generer_contexte_global()

        # Distribution déterministe, aussi équilibrée que possible entre les 3 types
        # (L, M, H) : 17/17/16 pour 50 machines (50 n'est pas un multiple de 3).
        cycle_types_machine = ["L", "M", "H"]
        qualite_type_machine = {mid: cycle_types_machine[i % 3] for i, mid in enumerate(self.config.ids_machines)}

        machine_dfs = []
        for mid in self.config.ids_machines:
            mtype = qualite_type_machine[mid]
            evolueur = EvolueurEtatMachine(mid, mtype, df_global, self.config)
            df_machine = evolueur.evoluer_etats()
            machine_dfs.append(df_machine)

        df_sensor = pd.concat(machine_dfs, ignore_index=True)

        df_final = df_sensor.merge(df_global, on="timestamp", how="left")
        if len(df_final) != len(df_sensor):
            raise ValueError(f"La fusion a modifié le nombre de lignes ! (Sensor: {len(df_sensor)}, Final: {len(df_final)})")

        agregateur = AgregateurDonnees(self.config)
        df_corrompu = agregateur.injecter_donnees_manquantes(df_final)
        agregateur.sauvegarder_parquet(df_corrompu)
        agregateur.inserer_duckdb(df_corrompu)

        logger.info(
            f"Dataset synthétique généré : {self.config.nb_machines} machines, "
            f"{self.config.duree_jours} jours, {self.config.intervalle_minutes} min/pas."
        )
        return df_corrompu


if __name__ == "__main__":
    orchestrateur = OrchestrateurDatasetSynthetique()
    jeu_donnees = orchestrateur.generer()
