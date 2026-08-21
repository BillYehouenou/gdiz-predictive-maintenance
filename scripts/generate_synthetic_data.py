"""
Module de génération de dataset synthétique pour la maintenance prédictive textile.

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


class SyntheticDataConfig:
    """Configuration centralisée pour la génération de données synthétiques."""

    global_seed = 42

    # Horizon temporel
    start_date = datetime(2024, 1, 1)
    duration_days = 730  # 2 ans — capture 2 cycles Harmattan + saisons des pluies
    interval_minutes = 30  # Pas de 30 min

    # Machines — 50 : au-delà, gain marginal sur la fiabilité des métriques de test
    # pour un coût ~x2 en temps/RAM (30 machines → marge ±11,5 pts sur le recall
    # événementiel mesuré sur le jeu de test ; 50 → ±8,7 pts ; 100 → ±6,2 pts).
    # Cf. docs/data-generation.md.
    nb_machines = 50
    machine_ids = [f"GDIZ_TEX_M_{i:02d}" for i in range(1, nb_machines + 1)]
    machine_type_quality = ["L", "M", "H"]

    # Usure — calibré pour ~4-8 pannes / machine / 2 ans (moyenne empirique : 6.6)
    base_wear_increment = 0.005
    initial_wear_range = (10, 70)

    # Fragilité individuelle de chaque machine — variabilité unité-à-unité de fabrication
    frailty_shape = 2.5

    # TWF (Tool Wear Failure) — chaque machine tire son propre seuil de rupture
    # La plupart des machines tiennent bien, certaines cassent dès 65-75 %, d'autres tiennent jusqu'à 95-99 %
    # Le hasard ne devient significatif qu'à l'approche de CE seuil personnel — pas d'un seuil universel — via une rampe lisse
    twf_threshold_min = 65.0
    twf_threshold_range = 34.0  # seuils personnels dans [65, 99]
    twf_threshold_beta_a = 4.0
    twf_threshold_beta_b = 2.0
    twf_ramp_band = 12.0  # largeur (en points d'usure) de la rampe avant le seuil personnel
    twf_ramp_exponent = 3.0
    # Recalibré (0.05 -> 0.08) pour compenser la dilution de part par les nouveaux
    # canaux HDF/OSF et par la maintenance préventive ci-dessous — cf.
    # docs/data-generation.md pour la méthode de calibration.
    twf_hazard_coeff = 0.08

    # HDF (Heat Dissipation Failure) — surchauffe soutenue (moyenne glissante ~4h), pas un pic instantané
    # Hasard nul sous le seuil, quadratique au-delà
    hdf_window_steps = 8  # 8 × 30 min = 4h
    hdf_overheat_threshold = 38.0  # °C au-delà de l'ambiant en moyenne sur la fenêtre choisie
    # Recalibré (0.0004 -> 0.00012) : la chaîne encrassement (fouling_heat_factor)
    # ajoute une source de chaleur supplémentaire, sans baisse du coefficient HDF
    # dominait les autres causes (jusqu'à 65 % des pannes lors du premier essai).
    hdf_hazard_coeff = 0.00012

    # OSF (Overstrain Failure) — surcharge mécanique : couple élevé combiné à une usure déjà significative.
    # Recalibré (0.00038 -> 0.0007) pour compenser la dilution par le mélange avec
    # bearing_wear (osf_bearing_weight ci-dessous).
    osf_hazard_coeff = 0.0007

    # PWF (Power Failure) — instabilité électrique routinière hors coupure franche
    # (le choc de redémarrage après blackout est géré séparément, cf. blackout_failure_sensitivity)
    # Recalibré (0.000024 -> 0.000015) : sa part croît mécaniquement quand les autres
    # causes baissent (risques concurrents), sans rapport avec un changement de PWF lui-même.
    pwf_hazard_coeff = 0.000015

    # RNF (Random Failure) — risque résiduel irréductible, par construction sans précurseur (miroir du RNF d'AI4I 2020)
    rnf_hazard_base = 0.000018

    # Dérive thermique additionnelle due au frottement d'un outil usé
    wear_heat_factor = 3.0  # °C additionnels max à usure = 100 %

    # Rampe terminale de dégradation (TWF) — au-delà du seuil personnel, l'usure peut
    # plafonner (capée à 100 %) alors que la panne n'a pas encore été tirée : sans ce
    # mécanisme, le hasard redevient un pur tirage mémoryless une fois le plafond
    # atteint, non prédictible depuis les capteurs. ramp_exposure_hours compte le temps
    # cumulé passé dans la bande de rampe et fait évoluer le hasard + la vibration/
    # chaleur même quand l'usure elle-même ne bouge plus.
    twf_ramp_max_exposure_hours = 240.0  # plafond (10 j) — au-delà, effet saturé
    twf_ramp_time_half_life_hours = 48.0  # le hasard TWF ~double tous les 2 jours passés dans la bande
    twf_chatter_rate = 0.022  # vibration additionnelle par heure d'exposition à la bande
    twf_ramp_heat_rate = 0.014  # °C additionnels par heure d'exposition à la bande

    # Encrassement (poussière → dissipation thermique réduite → HDF)
    fouling_base_rate = 0.06  # % d'encrassement gagné par pas, à dust=100 et sensibilité=1
    fouling_heat_factor = 1.8  # °C additionnels max à encrassement = 100 %
    fouling_reset_residual = 10.0  # % résiduel après une visite de maintenance (nettoyage partiel)

    # Maintenance préventive imparfaite — sans elle, les machines tournent toujours
    # jusqu'à la panne : les indicateurs précurseurs (usure, encrassement, roulements)
    # deviennent un prédicteur quasi certain de panne imminente, ce qui rend TWF
    # artificiellement facile. Une visite préventive (routine + risque, jamais
    # garantie) intercepte parfois une machine à risque avant qu'elle ne casse
    # réellement — le hasard de panne l'emporte la plupart du temps (imparfaite).
    preventive_base_prob = 0.00002  # composante routine, indépendante du risque
    # Recalibré (0.0004 -> 0.0001) : la première valeur interceptait trop de
    # trajectoires TWF (part TWF réduite de 32 % à 25 % des pannes, sans réaction à
    # une hausse du hasard TWF lui-même — même signal de risque pilotant les deux
    # mécanismes). Cf. docs/data-generation.md.
    preventive_risk_coeff = 0.0001  # composante risque, ∝ max(usure, encrassement, roulements)²
    preventive_duration_median_hours = 2.0
    preventive_duration_max_hours = 8.0
    preventive_wear_reduction = 0.5  # fraction d'usure/roulements retirée — service partiel, pas une remise à neuf

    # Second processus de dégradation indépendant de l'usure outil : usure des
    # roulements, pilotée par la vitesse/vibration plutôt que par la poussière ou le
    # couple — canal de dégradation distinct, contribue à OSF.
    bearing_wear_base_rate = 0.004
    bearing_vibration_coeff = 2.0
    osf_bearing_weight = 0.25  # part de bearing_wear_norm**3 dans le hasard OSF (reste : tool wear)

    # Parts de pannes visées par cause — cibles de calibration, pas une garantie stricte
    # (la réalisation dépend des coefficients ci-dessus, à vérifier empiriquement)
    failure_mode_shares = {"TWF": 0.40, "HDF": 0.20, "OSF": 0.15, "PWF": 0.20, "RNF": 0.05}

    # Thermique — inertie machine forte : 30 min ≈ 60 % du chemin vers T_cible
    coef_lissage_temp = 0.55
    friction_temp_factor = 25.0

    # Électrique et environnement
    # 220V = norme béninoise (réseau CEB/SBEE), pas les 230V harmonisés européens.
    nominal_voltage = 220.0
    voltage_variation_range = 0.15

    # Température atelier de référence par mois (°C), ancrée mi-mois puis interpolée
    # (voir generate_ambient_temperature) — calendrier bimodal réel du Sud-Bénin :
    # pic chaud fév-mars, creux fin juillet-août (petite saison sèche, remontée
    # d'eau froide atlantique), léger réchauffement sept-oct, Harmattan nov-mars.
    # Cf. docs/data-generation.md pour le détail et les sources.
    monthly_base_temp = {
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
    sensor_burst_nan_start_prob = 0.0024  # probabilité de démarrage d'une rafale, par pas/capteur
    sensor_burst_nan_duration_mean_steps = 3.0
    sensor_burst_nan_duration_sigma = 0.6
    sensor_burst_nan_duration_range = (1, 24)  # 30 min à 12h

    sensor_stuck_start_prob = 0.00025
    sensor_stuck_duration_mean_steps = 12.0
    sensor_stuck_duration_sigma = 1.0
    sensor_stuck_duration_range = (2, 96)  # 1h à 48h

    sensor_drift_start_prob = 0.00006
    sensor_drift_duration_mean_steps = 48.0
    sensor_drift_duration_sigma = 1.0
    sensor_drift_duration_range = (12, 200)  # 6h à 100h
    sensor_drift_max_pct_of_std = 0.5  # biais max atteint en fin de dérive, en fraction de l'écart-type du capteur

    # Chemins d'export
    output_dir = pathlib.Path("data/raw")
    parquet_file = output_dir / "gdiz_dataset.parquet"
    duckdb_file = output_dir / "gdiz_maintenance.db"


# A - Contexte exogène global


class ExogenousContextGenerator:
    """Génère la chronologie environnementale et électrique partagée par toute l'usine."""

    def __init__(self, config: SyntheticDataConfig = None):
        self.config = config or SyntheticDataConfig()
        np.random.seed(self.config.global_seed)
        self.num_samples = self._calculate_num_samples()
        logger.info(f"Contexte exogène initialisé: {self.num_samples} enregistrements à générer...")

    def _calculate_num_samples(self) -> int:
        return int(self.config.duration_days * 24 * 60 / self.config.interval_minutes)

    def generate_timestamps(self) -> pd.DatetimeIndex:
        return pd.date_range(start=self.config.start_date, periods=self.num_samples, freq=f"{self.config.interval_minutes}min")

    def generate_activity_level(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """Modélise un rythme d'activité industriel de 3 équipes en roulement 3x8."""
        hours = timestamps.hour.values
        minutes = timestamps.minute.values
        days = timestamps.dayofweek.values

        activity = np.zeros(len(timestamps))
        activity = np.where((hours >= 6) & (hours < 14), 0.92, activity)
        activity = np.where((hours >= 14) & (hours < 22), 0.86, activity)
        activity = np.where((hours >= 22) | (hours < 6), 0.55, activity)

        is_handover = (minutes < 20) & np.isin(hours, [6, 14, 22])
        activity = np.where(is_handover, activity * 0.75, activity)

        operational_noise = np.random.normal(0, 0.04, len(timestamps))
        activity = activity + operational_noise
        activity = np.where(days >= 5, 0.0, activity)
        return np.clip(activity, 0.0, 1.0)

    def generate_benin_seasons(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """
        Calendrier bimodal réel du Sud-Bénin (climat guinéen côtier) : deux saisons des
        pluies et deux saisons sèches, pas le cycle unimodal du Bénin septentrional.
            0 = grande saison des pluies    (mi-mars → mi-juillet)
            1 = petite saison sèche         (mi-juillet → août) — la plus fraîche de l'année
            2 = petite saison des pluies    (septembre → octobre)
            3 = grande saison sèche/Harmattan (novembre → mi-mars)
        Cf. docs/data-generation.md.
        """
        months = timestamps.month.values
        days = timestamps.day.values
        seasons = np.full(len(timestamps), 3, dtype=int)  # défaut : grande saison sèche

        is_grande_pluies = np.isin(months, [4, 5, 6]) | ((months == 3) & (days >= 15)) | ((months == 7) & (days < 15))
        is_petite_seche = ((months == 7) & (days >= 15)) | (months == 8)
        is_petite_pluies = np.isin(months, [9, 10])

        seasons[is_grande_pluies] = 0
        seasons[is_petite_seche] = 1
        seasons[is_petite_pluies] = 2
        return seasons

    def _monthly_anchor_temperature(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """
        Interpolation linéaire circulaire entre 12 ancrages mi-mois (config.monthly_base_temp) :
        remplace l'ancien saut en escalier (jusqu'à 4°C) au changement de mois par une
        transition continue, sans supposer une forme sinusoïdale simple (le cycle annuel
        du Sud-Bénin est bimodal, pas une sinusoïde unique).
        """
        anchors = self.config.monthly_base_temp
        mid_month_doy = np.array([15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349], dtype=float)
        anchor_temps = np.array([anchors[m] for m in range(1, 13)])
        # Périodicité : on prolonge d'un point avant janvier et un après décembre
        doy_ext = np.concatenate([[mid_month_doy[-1] - 365.0], mid_month_doy, [mid_month_doy[0] + 365.0]])
        temp_ext = np.concatenate([[anchor_temps[-1]], anchor_temps, [anchor_temps[0]]])
        doy = timestamps.dayofyear.values.astype(float)
        return np.interp(doy, doy_ext, temp_ext)

    def generate_ambient_temperature(self, timestamps: pd.DatetimeIndex, seasons: np.ndarray) -> np.ndarray:
        """
        Température de l'atelier combinant normale saisonnière lissée et cycle jour/nuit.
        Régulation industrielle : si T_brute > 38°C, la ventilation dissipe 70 % de l'excès.
        """
        hours = timestamps.hour.values

        daily_variation = 4.5 * np.sin(2 * np.pi * (hours - 8) / 24)
        base_temp = self._monthly_anchor_temperature(timestamps)
        # Air plus stable (moins de bruit) en saison sèche (1, 3) qu'en saison des pluies (0, 2)
        is_dry_ish = np.isin(seasons, [1, 3])
        noise = np.where(is_dry_ish, np.random.normal(0, 1.0, len(timestamps)), np.random.normal(0, 1.8, len(timestamps)))

        ambient_temp = base_temp + daily_variation + noise

        # Régulation/ventilation industrielle : dissipe 70 % de la chaleur excédentaire > 38°C
        ambient_temp = np.where(ambient_temp > 38.0, 38.0 + 0.30 * (ambient_temp - 38.0), ambient_temp)
        return np.clip(ambient_temp, 23.0, 42.0)

    # (p_start, p_stop) par saison — cf. generate_benin_seasons pour le code des 4 saisons.
    # La petite saison des pluies (2) est en moyenne moins intense que la grande (0) ;
    # la petite saison sèche (1) n'est pas aussi aride que l'Harmattan (3).
    RAIN_TRANSITION_PROBS = {
        0: (0.12, 0.40),  # grande saison des pluies
        1: (0.02, 0.65),  # petite saison sèche
        2: (0.08, 0.45),  # petite saison des pluies
        3: (0.006, 0.75),  # grande saison sèche / Harmattan
    }

    def generate_rain_events(self, seasons: np.ndarray) -> np.ndarray:
        """
        Génère des épisodes de pluie cohérents via chaîne de Markov.
        Probabilités recalibrées pour un pas de 30 min (≈ 6× les valeurs 5 min).
        """
        num_samples = len(seasons)
        rain_flag = np.zeros(num_samples, dtype=int)

        for i in range(1, num_samples):
            p_start, p_stop = self.RAIN_TRANSITION_PROBS[int(seasons[i])]

            if rain_flag[i - 1] == 1:
                rain_flag[i] = 1 if np.random.rand() > p_stop else 0
            else:
                rain_flag[i] = 1 if np.random.rand() < p_start else 0

        return rain_flag

    def generate_humidity(self, ambient_temp: np.ndarray, rain_flag: np.ndarray) -> np.ndarray:
        base_humidity = 86 - (ambient_temp - 30) * 2.3
        rain_effect = rain_flag * 9.5
        noise = np.random.normal(0, 3.0, len(ambient_temp))
        return np.clip(base_humidity + rain_effect + noise, 45, 98)

    def generate_dust_concentration(self, timestamps: pd.DatetimeIndex, seasons: np.ndarray, rain_flag: np.ndarray) -> np.ndarray:
        """Concentration de poussière : Effet Harmattan majeur en saison sèche, lavée par la pluie."""
        num_samples = len(seasons)
        months = timestamps.month.values
        is_harmattan = np.isin(months, [12, 1])
        is_dry_ish = np.isin(seasons, [1, 3])  # petite + grande saison sèche

        shape = np.select(condlist=[is_harmattan, is_dry_ish], choicelist=[5.0, 2.5], default=1.5)
        scale = np.select(condlist=[is_harmattan, is_dry_ish], choicelist=[30.0, 12.0], default=6.0)

        dust = np.random.gamma(shape, scale, num_samples)
        dust = np.where(rain_flag == 1, dust * 0.15, dust)
        return np.clip(dust, 5, 600)

    def generate_voltage_level(self, activity_level: np.ndarray) -> np.ndarray:
        base_voltage = self.config.nominal_voltage
        load_variation = -15 * activity_level
        noise = np.random.normal(0, 2.5, len(activity_level))
        return base_voltage + load_variation + noise

    def generate_voltage_stability(self, voltage: np.ndarray) -> np.ndarray:
        deviation = np.abs(voltage - self.config.nominal_voltage)
        stability = 100 * np.exp(-deviation / 18)
        noise = np.random.normal(0, 1.5, len(voltage))
        return np.clip(stability + noise, 0, 100)

    def generate_power_loss_indicator(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """
        Simule des délestages SBEE et coupures de secteur.
        Probabilités recalibrées pour un pas de 30 min.
        Réseau volontairement clément (hypothèse : alimentation dédiée GDIZ, cf.
        docs/data-generation.md) mais légèrement durci par rapport à la version
        précédente : coupures en moyenne ~67 min (continuation = 0.55), toujours
        loin des 10-15h que peut atteindre une coupure SBEE non dédiée.
        """
        hours = timestamps.hour.values
        num_samples = len(timestamps)

        base_prob = 0.0006  # ≈ 6× la valeur 5 min (0.0001)
        peak_hours = np.isin(hours, [11, 12, 13, 14, 19, 20, 21, 22])
        failure_probability = np.where(peak_hours, base_prob * 8, base_prob)

        power_loss = (np.random.random(num_samples) < failure_probability).astype(int)

        # E[durée] = 1/(1-p) pas ≈ 2.2 pas ≈ 67 min (contre ~50 min avant durcissement)
        for i in range(1, num_samples):
            if power_loss[i - 1] == 1 and np.random.random() < 0.55:
                power_loss[i] = 1

        return power_loss

    def generate_global_context(self) -> pd.DataFrame:
        timestamps = self.generate_timestamps()
        seasons = self.generate_benin_seasons(timestamps)
        rain_flag = self.generate_rain_events(seasons)
        ambient_temp = self.generate_ambient_temperature(timestamps, seasons)
        humidity = self.generate_humidity(ambient_temp, rain_flag)
        dust = self.generate_dust_concentration(timestamps, seasons, rain_flag)
        activity_level = self.generate_activity_level(timestamps)
        voltage = self.generate_voltage_level(activity_level)
        voltage_stability = self.generate_voltage_stability(voltage)
        power_loss = self.generate_power_loss_indicator(timestamps)

        df_global = pd.DataFrame(
            {
                "timestamp": timestamps,
                "activity_level": activity_level,
                "benin_season": seasons,
                "ambient_temperature": ambient_temp,
                "rain_flag": rain_flag,
                "humidity": humidity,
                "dust_concentration": dust,
                "voltage_level": voltage,
                "voltage_stability": voltage_stability,
                "power_loss_indicator": power_loss,
            }
        )
        logger.info(f"Contexte exogène généré avec succès ({df_global.shape[0]} enregistrements)")
        return df_global


# B - Moteur d'évolution chronologique par machine


class MachineStateEvolver:
    """Gère l'évolution d'état chronologique d'une machine (Dépendance physique N-1)."""

    def __init__(
        self,
        machine_id: str,
        machine_type: str,
        df_global: pd.DataFrame,
        config: SyntheticDataConfig = None,
        override_params: dict = None,
    ):
        self.machine_id = machine_id
        self.machine_type = machine_type
        self.df_global = df_global
        self.config = config or SyntheticDataConfig()

        # RNG isolé par machine — évite toute pollution croisée des flux aléatoires
        # Hash déterministe (md5) plutôt que hash() builtin : hash() est salé
        # aléatoirement par process (PYTHONHASHSEED) sauf s'il est fixé explicitement,
        # ce qui casserait la reproductibilité entre deux exécutions.
        machine_hash = int(hashlib.md5(machine_id.encode("utf-8")).hexdigest(), 16) % 10000
        machine_seed = self.config.global_seed + machine_hash
        self.rng = np.random.default_rng(machine_seed)

        # Fragilité individuelle (frailty model) — tirée une fois par machine,
        # n'affecte que le hasard de panne TWF, pas la vitesse d'usure elle-même.
        shape = self.config.frailty_shape
        self.frailty = float(self.rng.gamma(shape, 1.0 / shape))

        # Seuil de rupture personnel (TWF) — variabilité unité-à-unité de fabrication :
        # certaines machines cassent dès 65-75 % d'usure, d'autres tiennent jusqu'à 95-99 %.
        self.twf_threshold = self.config.twf_threshold_min + self.config.twf_threshold_range * self.rng.beta(
            self.config.twf_threshold_beta_a, self.config.twf_threshold_beta_b
        )

        self.num_samples = len(df_global)
        self.ambient_array = df_global["ambient_temperature"].values
        self.dust_array = df_global["dust_concentration"].values
        self.stability_array = df_global["voltage_stability"].values
        self.voltage_array = df_global["voltage_level"].values
        self.activity_level_array = df_global["activity_level"].values
        self.power_loss_array = df_global["power_loss_indicator"].values
        self._set_machine_parameters()
        if override_params:
            self.params.update(override_params)
        self._init_state_arrays()

    def _set_machine_parameters(self):
        type_params = {
            "L": {  # Entrée de gamme / Rustique — aucune protection électronique
                "wear_multiplier": 1.3,
                "temp_target_offset": 3.0,
                "vibration_noise": 0.25,
                "dust_sensitivity": 1.5,
                "voltage_sensitivity": 0.5,
                "power_loss_restart_malus": 0.8,
                # Coupure réseau : le choc mécanique au redémarrage est violent pour L
                "blackout_failure_sensitivity": 1.0,
            },
            "M": {  # Milieu de gamme / Standard
                "wear_multiplier": 1.0,
                "temp_target_offset": 0.0,
                "vibration_noise": 0.15,
                "dust_sensitivity": 1.0,
                "voltage_sensitivity": 1.0,
                "power_loss_restart_malus": 0.4,
                "blackout_failure_sensitivity": 0.8,
            },
            "H": {  # Haut de gamme / Précision Électronique + Onduleur UPS industriel
                "wear_multiplier": 0.7,
                "temp_target_offset": -2.0,
                "vibration_noise": 0.08,
                "dust_sensitivity": 0.4,
                # voltage_sensitivity conservé pour la physique (usure, chaleur) :
                # l'UPS ne supprime pas les effets physiques, il protège des pannes.
                "voltage_sensitivity": 2.0,
                "power_loss_restart_malus": 0.2,
                # Coupure réseau : l'UPS amortit la surtension → panne sur blackout rare
                "blackout_failure_sensitivity": 0.2,
                # Instabilité routinière : l'onduleur absorbe les creux < blackout total
                # sensibilité effective aux pannes électriques spontanées quasi nulle
                "ups_elec_sensitivity": 0.1,
            },
        }
        self.params = type_params[self.machine_type]

    def _init_state_arrays(self):
        self.tool_wear = np.zeros(self.num_samples)
        self.process_temperature = np.zeros(self.num_samples)
        self.rotational_speed = np.zeros(self.num_samples)
        self.torque = np.zeros(self.num_samples)
        self.vibration = np.zeros(self.num_samples)
        self.failure = np.zeros(self.num_samples, dtype=int)
        self.failure_mode = np.full(self.num_samples, "", dtype=object)

        # Encrassement (point 7) et usure des roulements (point 11) : mêmes limites
        # 0-100 % que tool_wear, mais accumulation et remise à neuf indépendantes.
        self.fouling = np.zeros(self.num_samples)
        self.bearing_wear = np.zeros(self.num_samples)
        # Temps cumulé (h) passé dans la bande de rampe TWF (point 8) — remis à zéro
        # à chaque visite de maintenance, quelle que soit la cause de la panne.
        self.ramp_exposure_hours = np.zeros(self.num_samples)

        self.tool_wear[0] = self.rng.uniform(*self.config.initial_wear_range)
        self.process_temperature[0] = self.ambient_array[0] + 5.0

    def _update_physics(self, idx: int, restarted_from_blackout: bool = False):
        """Calcule les indicateurs physiques de la machine lorsqu'elle est en fonctionnement."""
        activity_idx = self.activity_level_array[idx]
        stability_pct = self.stability_array[idx] / 100.0
        voltage_deviation = max(0, self.config.nominal_voltage - self.voltage_array[idx])

        # 1. Vitesse & Couple — découplés : dans un atelier textile réel, la vitesse
        # est gouvernée par le moteur et reste quasi constante une fois la machine
        # lancée ; c'est la CHARGE (couple) qui varie avec l'activité/production, pas
        # le régime moteur. (Avant : les deux étaient pilotés par activity_level,
        # corrélation mesurée +0.94 — cf. docs/data-generation.md.)
        base_speed = 1500.0
        self.rotational_speed[idx] = base_speed * (0.97 + 0.03 * stability_pct)
        self.rotational_speed[idx] += self.rng.normal(0, 12)
        self.rotational_speed[idx] = max(0, self.rotational_speed[idx])

        base_torque = 45.0
        self.torque[idx] = base_torque * activity_idx * (1.3 - 0.3 * stability_pct)
        self.torque[idx] += self.rng.normal(0, 3)
        self.torque[idx] = max(0, self.torque[idx])

        # 2. Usure cumulative (outil)
        wear_prev = self.tool_wear[idx - 1]
        if restarted_from_blackout:
            wear_prev = min(wear_prev + self.params["power_loss_restart_malus"], 100)

        dust_factor = 1.0 + 0.004 * self.dust_array[idx] * self.params["dust_sensitivity"]
        voltage_factor = 1.0 + 0.6 * (1.0 - stability_pct) * self.params["voltage_sensitivity"]
        wear_increment = self.config.base_wear_increment * self.params["wear_multiplier"]
        wear_increment *= dust_factor * voltage_factor
        self.tool_wear[idx] = min(wear_prev + wear_increment, 100)

        # 2b. Encrassement — accumulation indépendante liée à l'exposition à la
        # poussière, remise à un résidu (fouling_reset_residual) à chaque visite de
        # maintenance (cf. _handle_downtime_phase). Alimente la chaleur, pas l'usure
        # outil : deux canaux physiques distincts vers HDF vs TWF.
        fouling_increment = self.config.fouling_base_rate * (self.dust_array[idx] / 100.0) * self.params["dust_sensitivity"]
        self.fouling[idx] = min(self.fouling[idx - 1] + fouling_increment, 100.0)

        # 2c. Usure des roulements — second processus de dégradation, indépendant de
        # l'usure outil : piloté par la vitesse (fatigue mécanique), pas la poussière
        # ni le couple. Remis à zéro seulement quand OSF est la cause réparée
        # (cf. _handle_downtime_phase).
        speed_ratio = self.rotational_speed[idx] / base_speed
        bearing_increment = self.config.bearing_wear_base_rate * (speed_ratio**2) * self.params["wear_multiplier"]
        self.bearing_wear[idx] = min(self.bearing_wear[idx - 1] + bearing_increment, 100.0)

        # 2d. Rampe terminale — temps cumulé passé dans la bande de rampe TWF (autour
        # du seuil personnel). Continue de s'accumuler même quand tool_wear plafonne à
        # 100 % : sans ça, le hasard de panne redevient un pur tirage mémoryless une
        # fois l'usure saturée, invisible depuis les capteurs (cf. point 8 — condition
        # nécessaire pour tout modèle de survie futur).
        in_ramp_band = self.tool_wear[idx] >= (self.twf_threshold - self.config.twf_ramp_band)
        if in_ramp_band:
            step_hours = self.config.interval_minutes / 60.0
            self.ramp_exposure_hours[idx] = min(
                self.ramp_exposure_hours[idx - 1] + step_hours, self.config.twf_ramp_max_exposure_hours
            )
        else:
            self.ramp_exposure_hours[idx] = 0.0

        # 3. Température Machine
        temp_prev = self.process_temperature[idx - 1]
        electrical_heat = (voltage_deviation * 0.15) * self.params["voltage_sensitivity"]
        # Frottement accru par l'usure : dérive thermique supplémentaire, corrélée à
        # la vibration — donne deux indicateurs précurseurs conjoints pour TWF.
        wear_heat = self.config.wear_heat_factor * (self.tool_wear[idx] / 100.0)
        # Encrassement : dissipation thermique réduite par la poussière accumulée.
        fouling_heat = self.config.fouling_heat_factor * (self.fouling[idx] / 100.0)
        # Rampe terminale : chaleur additionnelle qui continue de croître avec le
        # temps passé en zone critique, même quand tool_wear a plafonné.
        ramp_heat = self.config.twf_ramp_heat_rate * self.ramp_exposure_hours[idx]
        target_temp = (
            self.ambient_array[idx]
            + (self.config.friction_temp_factor * (self.torque[idx] / 35.0))
            + self.params["temp_target_offset"]
            + electrical_heat
            + wear_heat
            + fouling_heat
            + ramp_heat
        )
        alpha = self.config.coef_lissage_temp
        self.process_temperature[idx] = alpha * target_temp + (1 - alpha) * temp_prev

        # 4. Vibrations
        wear_norm = self.tool_wear[idx] / 100.0
        bearing_norm = self.bearing_wear[idx] / 100.0
        base_vibration = 0.4 * (self.rotational_speed[idx] / base_speed) + 0.4 * (1.0 - stability_pct)
        if restarted_from_blackout:
            base_vibration += self.rng.uniform(3.0, 5.0) * self.params["wear_multiplier"]
        wear_effect = 4.5 * (wear_norm**2)
        bearing_effect = self.config.bearing_vibration_coeff * (bearing_norm**2)
        # Broutement d'outil (chatter) : s'amplifie avec le temps passé en zone
        # critique — signal observable qui continue de monter même quand wear_effect
        # a plafonné (point 8).
        chatter_effect = self.config.twf_chatter_rate * self.ramp_exposure_hours[idx]
        noise = self.rng.normal(0, self.params["vibration_noise"])
        self.vibration[idx] = np.clip(base_vibration + wear_effect + bearing_effect + chatter_effect + noise, 0.05, 15.0)

    def _calculate_failure_hazards(self, idx: int) -> dict:
        """
        Évalue le hasard de panne par cause au pas actuel (modèle à risques concurrents,
        façon AI4I 2020 : TWF, HDF, OSF, PWF, RNF). Chaque cause a sa propre signature
        précurseur ; aucune ne repose sur un seuil dur déterministe.
        """
        wear = self.tool_wear[idx]
        wear_norm = wear / 100.0
        torque = self.torque[idx]
        stability_pct = self.stability_array[idx] / 100.0

        # TWF — rampe lisse autour du seuil de rupture personnel de la machine
        # (self.twf_threshold, tiré une fois à l'initialisation) : pas de seuil
        # universel, étalement réel du moment de panne entre machines fragiles et
        # robustes. La fragilité (self.frailty) module l'intensité de la rampe.
        band = self.config.twf_ramp_band
        excess = max(0.0, (wear - (self.twf_threshold - band)) / band)
        # Facteur temporel : le hasard continue de croître avec le temps cumulé passé
        # dans la bande de rampe, même quand `excess` a plafonné parce que l'usure est
        # saturée à 100 % — sans ça, une fois saturée, la panne redevient un pur
        # tirage mémoryless (cf. ramp_exposure_hours dans _update_physics, point 8).
        time_factor = 1.0 + self.ramp_exposure_hours[idx] / self.config.twf_ramp_time_half_life_hours
        twf = (
            self.config.twf_hazard_coeff
            * (excess**self.config.twf_ramp_exponent)
            * time_factor
            * self.frailty
            * self.params["wear_multiplier"]
        )

        # HDF — surchauffe soutenue sur une fenêtre glissante de ~4h, pas un pic
        # instantané : la dissipation thermique insuffisante est un phénomène lent.
        window = min(idx, self.config.hdf_window_steps)
        overheat = self.process_temperature[idx - window : idx + 1] - self.ambient_array[idx - window : idx + 1]
        sustained_overheat = max(0.0, float(overheat.mean()) - self.config.hdf_overheat_threshold)
        hdf = self.config.hdf_hazard_coeff * sustained_overheat**2

        # OSF — surcharge mécanique : couple élevé combiné à une usure déjà
        # significative. wear_norm**3 supprime naturellement le hasard à usure
        # faible, sans seuil dur explicite. Combine usure outil et usure des
        # roulements (second processus de dégradation indépendant, point 11) — pas
        # seulement l'usure outil, sinon OSF ne fait que dupliquer le signal TWF.
        bearing_norm = self.bearing_wear[idx] / 100.0
        osf_wear_term = (1.0 - self.config.osf_bearing_weight) * wear_norm**3 + self.config.osf_bearing_weight * bearing_norm**3
        osf = self.config.osf_hazard_coeff * (torque / 45.0) * osf_wear_term

        # PWF — instabilité électrique routinière hors coupure franche (le choc
        # de redémarrage après blackout est géré séparément dans evolve_states).
        elec_sensitivity = self.params.get("ups_elec_sensitivity", self.params["voltage_sensitivity"])
        pwf = self.config.pwf_hazard_coeff * max(0.0, 1.0 - stability_pct) * elec_sensitivity

        # RNF — risque résiduel irréductible, par construction sans précurseur.
        rnf = self.config.rnf_hazard_base

        return {"TWF": twf, "HDF": hdf, "OSF": osf, "PWF": pwf, "RNF": rnf}

    def _handle_downtime_phase(self, failure_idx: int) -> int:
        """
        Met la machine en sommeil pour maintenance suite à une panne.
        Durée générée via loi Log-Normale : médiane ≈ 3h, queue jusqu'à 48h.
        Refroidissement passif accéléré : alpha = 0.70 (retour rapide à l'ambiant en 30 min).
        """
        alpha_refroidissement = 0.70

        # Log-Normale : médiane = exp(ln(3)) = 3h, σ=1.2 → P99 ≈ 48h naturellement
        duration_hours = float(self.rng.lognormal(mean=np.log(3.0), sigma=1.2))
        duration_hours = float(np.clip(duration_hours, 0.5, 48.0))
        num_steps = int(np.ceil(duration_hours * 60 / self.config.interval_minutes))

        start_downtime = failure_idx + 1
        end_downtime = min(start_downtime + num_steps, self.num_samples)

        for idx in range(start_downtime, end_downtime):
            self.rotational_speed[idx] = 0.0
            self.torque[idx] = 0.0
            self.vibration[idx] = np.abs(self.rng.normal(0, 0.01))
            self.failure[idx] = 0
            self.tool_wear[idx] = self.tool_wear[idx - 1]
            self.fouling[idx] = self.fouling[idx - 1]
            self.bearing_wear[idx] = self.bearing_wear[idx - 1]
            self.ramp_exposure_hours[idx] = self.ramp_exposure_hours[idx - 1]

            # Refroidissement passif accéléré : en 30 min, la machine retombe quasi à l'ambiant
            self.process_temperature[idx] = (
                alpha_refroidissement * self.ambient_array[idx] + (1 - alpha_refroidissement) * self.process_temperature[idx - 1]
            )

        if end_downtime < self.num_samples:
            cause = self.failure_mode[failure_idx]
            # Remise à neuf de l'outil — seulement si la panne réparée est l'outil
            # lui-même (TWF). Les autres causes réparent un autre composant : l'usure
            # de l'outil continue de s'accumuler sans être remise à zéro, sinon les
            # pannes non-mécaniques interrompraient indéfiniment la montée en usure et
            # TWF ne se déclencherait jamais.
            if cause == "TWF":
                self.tool_wear[end_downtime - 1] = 0.0
            # Remise à neuf des roulements — seulement si OSF est la cause réparée
            # (même logique que TWF/tool_wear, second processus indépendant).
            if cause == "OSF":
                self.bearing_wear[end_downtime - 1] = 0.0
            # Toute visite de maintenance inclut un entretien de base, quelle que soit
            # la cause : nettoyage partiel (encrassement ramené au plus à un résidu,
            # jamais remonté artificiellement) et contrôle de sécurité (zone de rampe
            # réinitialisée).
            self.fouling[end_downtime - 1] = min(self.fouling[end_downtime - 1], self.config.fouling_reset_residual)
            self.ramp_exposure_hours[end_downtime - 1] = 0.0

        return end_downtime

    def _preventive_maintenance_prob(self, idx: int) -> float:
        """
        Probabilité qu'une visite de maintenance préventive intercepte la machine à
        ce pas — composante routine (calendaire, indépendante du risque) + composante
        risque (usure/encrassement/roulements). Reste délibérément faible devant le
        hasard de panne réel : l'interception est possible, pas garantie (imparfaite).
        """
        risk_score = max(self.tool_wear[idx], self.fouling[idx], self.bearing_wear[idx]) / 100.0
        return self.config.preventive_base_prob + self.config.preventive_risk_coeff * risk_score**2

    def _handle_preventive_maintenance_phase(self, event_idx: int) -> int:
        """
        Visite de maintenance préventive : pas une panne (failure reste à 0, aucun
        failure_mode enregistré), plus courte qu'une réparation, et surtout
        imparfaite — réduit l'usure/les roulements sans les remettre à neuf,
        contrairement à une réparation de panne qui répare le composant en cause.
        """
        duration_hours = float(self.rng.lognormal(mean=np.log(self.config.preventive_duration_median_hours), sigma=0.6))
        duration_hours = float(np.clip(duration_hours, 0.5, self.config.preventive_duration_max_hours))
        num_steps = int(np.ceil(duration_hours * 60 / self.config.interval_minutes))

        start = event_idx + 1
        end = min(start + num_steps, self.num_samples)
        alpha_refroidissement = 0.70

        for idx in range(start, end):
            self.rotational_speed[idx] = 0.0
            self.torque[idx] = 0.0
            self.vibration[idx] = np.abs(self.rng.normal(0, 0.01))
            self.failure[idx] = 0
            self.tool_wear[idx] = self.tool_wear[idx - 1]
            self.fouling[idx] = self.fouling[idx - 1]
            self.bearing_wear[idx] = self.bearing_wear[idx - 1]
            self.ramp_exposure_hours[idx] = self.ramp_exposure_hours[idx - 1]
            self.process_temperature[idx] = (
                alpha_refroidissement * self.ambient_array[idx] + (1 - alpha_refroidissement) * self.process_temperature[idx - 1]
            )

        if end < self.num_samples:
            # Service partiel — réduit sans remettre à neuf (contrairement à une
            # réparation de panne, cf. _handle_downtime_phase).
            self.tool_wear[end - 1] *= 1.0 - self.config.preventive_wear_reduction
            self.bearing_wear[end - 1] *= 1.0 - self.config.preventive_wear_reduction
            self.fouling[end - 1] = min(self.fouling[end - 1], self.config.fouling_reset_residual)
            self.ramp_exposure_hours[end - 1] = 0.0

        return end

    def evolve_states(self) -> pd.DataFrame:
        """Exécute la boucle d'évolution séquentielle sur la chronologie de la machine."""
        maintenance_end_idx = 0

        for idx in range(1, self.num_samples):
            if idx < maintenance_end_idx:
                continue

            # Interception des délestages et coupures de courant
            if self.power_loss_array[idx] == 1:
                self.rotational_speed[idx] = 0.0
                self.torque[idx] = 0.0
                self.vibration[idx] = np.abs(self.rng.normal(0, 0.01))
                self.tool_wear[idx] = self.tool_wear[idx - 1]
                self.fouling[idx] = self.fouling[idx - 1]
                self.bearing_wear[idx] = self.bearing_wear[idx - 1]
                self.ramp_exposure_hours[idx] = self.ramp_exposure_hours[idx - 1]
                self.failure[idx] = 0

                alpha = self.config.coef_lissage_temp
                self.process_temperature[idx] = alpha * self.ambient_array[idx] + (1 - alpha) * self.process_temperature[idx - 1]

                if self.power_loss_array[idx - 1] == 0:
                    # L=4%, M=3.2%, H=0.8% — H protégé par l'UPS au choc de coupure
                    electric_shock_prob = 0.04 * self.params["blackout_failure_sensitivity"]
                    if self.rng.random() < electric_shock_prob:
                        self.failure[idx] = 1
                        self.failure_mode[idx] = "PWF"
                        maintenance_end_idx = self._handle_downtime_phase(idx)
                continue

            restarted_from_blackout = self.power_loss_array[idx - 1] == 1

            # Arrêts planifiés (week-end, activité nulle)
            if self.activity_level_array[idx] == 0.0:
                self.rotational_speed[idx] = 0.0
                self.torque[idx] = 0.0
                self.vibration[idx] = np.abs(self.rng.normal(0, 0.01))
                self.tool_wear[idx] = self.tool_wear[idx - 1]
                self.fouling[idx] = self.fouling[idx - 1]
                self.bearing_wear[idx] = self.bearing_wear[idx - 1]
                self.ramp_exposure_hours[idx] = self.ramp_exposure_hours[idx - 1]
                self.failure[idx] = 0

                alpha = self.config.coef_lissage_temp
                self.process_temperature[idx] = alpha * self.ambient_array[idx] + (1 - alpha) * self.process_temperature[idx - 1]
                continue

            # Régime de fonctionnement normal
            self._update_physics(idx, restarted_from_blackout=restarted_from_blackout)

            hazards = self._calculate_failure_hazards(idx)
            total_hazard = sum(hazards.values())
            failure_prob = min(total_hazard, 1.0)
            if self.rng.random() < failure_prob:
                self.failure[idx] = 1
                modes = list(hazards.keys())
                weights = np.array(list(hazards.values()))
                mode_probs = weights / total_hazard if total_hazard > 0 else np.full(len(modes), 1.0 / len(modes))
                self.failure_mode[idx] = self.rng.choice(modes, p=mode_probs)
                maintenance_end_idx = self._handle_downtime_phase(idx)
            elif self.rng.random() < self._preventive_maintenance_prob(idx):
                # Maintenance préventive imparfaite : intercepte parfois une machine à
                # risque avant la panne réelle — cf. _preventive_maintenance_prob.
                maintenance_end_idx = self._handle_preventive_maintenance_phase(idx)

        return pd.DataFrame(
            {
                "machine_id": self.machine_id,
                "machine_type": self.machine_type,
                "timestamp": self.df_global["timestamp"],
                "rotational_speed": self.rotational_speed,
                "torque": self.torque,
                "tool_wear": self.tool_wear,
                "fouling": self.fouling,
                "bearing_wear": self.bearing_wear,
                "process_temperature": self.process_temperature,
                "vibration": self.vibration,
                "target": self.failure,
                "failure_mode": self.failure_mode,
            }
        )


# C - Agrégation, masquage IoT et persistence des données


class DataAggregator:
    """Combine les historiques des machines et orchestre la persistence des données."""

    def __init__(self, config: SyntheticDataConfig = None):
        self.config = config or SyntheticDataConfig()

    def _apply_bursts(
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

    def _apply_drift(self, values: np.ndarray, scale: float) -> tuple[np.ndarray, int]:
        """
        Dérive de calibration : biais qui croît linéairement (rampe 0 → max) sur la
        durée de l'événement, signe aléatoire, amplitude proportionnelle à l'écart-type
        du capteur — pas de valeur manquante, un biais silencieux, plus difficile à
        repérer qu'un NaN.
        """
        n = len(values)
        out = values.copy()
        starts = np.flatnonzero(np.random.random(n) < self.config.sensor_drift_start_prob)
        dmin, dmax = self.config.sensor_drift_duration_range
        affected = 0
        for s in starts:
            duration = int(
                np.ceil(
                    np.clip(
                        np.random.lognormal(
                            np.log(self.config.sensor_drift_duration_mean_steps), self.config.sensor_drift_duration_sigma
                        ),
                        dmin,
                        dmax,
                    )
                )
            )
            end = min(s + duration, n)
            sign = 1.0 if np.random.random() < 0.5 else -1.0
            ramp = np.linspace(0.0, 1.0, end - s)
            out[s:end] += sign * scale * self.config.sensor_drift_max_pct_of_std * ramp
            affected += end - s
        return out, affected

    def inject_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Simule des défaillances de capteurs IoT réalistes plutôt qu'un MCAR uniforme
        (une cellule isolée sur deux sans lien avec les autres, le cas le plus facile
        à imputer) : rafales de paquets perdus, capteurs figés, dérive de calibration.
        Généré indépendamment par (machine, capteur) — pas par cellule.
        """
        df_copy = df.copy()
        sensor_cols = [
            "tool_wear",
            "process_temperature",
            "rotational_speed",
            "torque",
            "vibration",
            "ambient_temperature",
            "humidity",
            "dust_concentration",
        ]
        sensor_cols = [col for col in sensor_cols if col in df_copy.columns]

        if not sensor_cols or "machine_id" not in df_copy.columns or df_copy.empty:
            return df_copy

        # Extraction explicite en float → corruption → réassignation propre. copy=True :
        # sous pandas 3.x (copy-on-write par défaut), to_numpy() sans copie explicite
        # peut renvoyer un tableau read-only et faire échouer l'affectation ci-dessous.
        sensor_data = df_copy[sensor_cols].to_numpy(dtype=float, copy=True)

        n_nan = n_stuck = n_drift = 0
        for positions in df_copy.groupby("machine_id", sort=False).indices.values():
            positions = np.asarray(positions)
            for c in range(len(sensor_cols)):
                values = sensor_data[positions, c]
                scale = float(np.nanstd(values)) or 1.0

                values, affected = self._apply_drift(values, scale)
                n_drift += affected

                values, affected = self._apply_bursts(
                    values,
                    self.config.sensor_stuck_start_prob,
                    self.config.sensor_stuck_duration_mean_steps,
                    self.config.sensor_stuck_duration_sigma,
                    self.config.sensor_stuck_duration_range,
                    mode="stuck",
                )
                n_stuck += affected

                values, affected = self._apply_bursts(
                    values,
                    self.config.sensor_burst_nan_start_prob,
                    self.config.sensor_burst_nan_duration_mean_steps,
                    self.config.sensor_burst_nan_duration_sigma,
                    self.config.sensor_burst_nan_duration_range,
                    mode="nan",
                )
                n_nan += affected

                sensor_data[positions, c] = values

        df_copy[sensor_cols] = sensor_data
        logger.info(
            f"Défaillance capteurs : {n_nan} valeurs manquantes (rafales), {n_stuck} figées, {n_drift} en dérive de calibration."
        )
        return df_copy

    def save_parquet(self, df: pd.DataFrame):
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.config.parquet_file, compression="snappy", index=False)
        size_mb = self.config.parquet_file.stat().st_size / (1024**2)
        logger.info(f"Fichier {self.config.parquet_file} sauvegardé ({size_mb:.2f} MB)")

    def insert_to_duckdb(self, df: pd.DataFrame):
        db_path = pathlib.Path(self.config.duckdb_file).resolve()
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
                conn.execute(f"CREATE INDEX idx_machine_time ON {table_name}(machine_id, timestamp)")
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


class SyntheticDatasetOrchestrator:
    """Pilote suprême du pipeline d'ingénierie des données synthétiques."""

    def __init__(self, random_seed: int | None = None):
        self.config = SyntheticDataConfig()
        if random_seed is not None:
            self.config.global_seed = random_seed
        # Pas de np.random.seed() ici : ExogenousContextGenerator.__init__ reseed
        # déjà avec self.config.global_seed juste après — l'appel ici était redondant
        # et source de confusion sur la source de vérité du seed.

    def generate(self) -> pd.DataFrame:
        exo_gen = ExogenousContextGenerator(self.config)
        df_global = exo_gen.generate_global_context()

        # Distribution déterministe, aussi équilibrée que possible entre les 3 types
        # (L, M, H) : 17/17/16 pour 50 machines (50 n'est pas un multiple de 3).
        machine_types_cycle = ["L", "M", "H"]
        machine_type_quality = {mid: machine_types_cycle[i % 3] for i, mid in enumerate(self.config.machine_ids)}

        machine_dfs = []
        for mid in self.config.machine_ids:
            mtype = machine_type_quality[mid]
            evolver = MachineStateEvolver(mid, mtype, df_global, self.config)
            df_machine = evolver.evolve_states()
            machine_dfs.append(df_machine)

        df_sensor = pd.concat(machine_dfs, ignore_index=True)

        df_final = df_sensor.merge(df_global, on="timestamp", how="left")
        if len(df_final) != len(df_sensor):
            raise ValueError(f"La fusion a modifié le nombre de lignes ! (Sensor: {len(df_sensor)}, Final: {len(df_final)})")

        aggregator = DataAggregator(self.config)
        df_corrupted = aggregator.inject_missing_data(df_final)
        aggregator.save_parquet(df_corrupted)
        aggregator.insert_to_duckdb(df_corrupted)

        logger.info(
            f"Dataset synthétique généré : {self.config.nb_machines} machines, "
            f"{self.config.duration_days} jours, {self.config.interval_minutes} min/pas."
        )
        return df_corrupted


if __name__ == "__main__":
    orchestrator = SyntheticDatasetOrchestrator()
    dataset = orchestrator.generate()
