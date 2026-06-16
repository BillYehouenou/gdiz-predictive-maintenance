"""
Module de génération de dataset synthétique pour la maintenance prédictive textile.

Architecture hybride:
- Étape A: Contexte exogène global (variables macro-environnementales)
- Étape B: Moteur d'évolution séquentiel (boucle par machine avec états N-1)
- Étape C: Agrégation en DataFrame unique et ingestion DuckDB

Framework académique: AI4I 2020 adapté à l'écosystème textile de la GDIZ.
"""

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
    interval_minutes = 30  # Pas de 30 min : budget calcul × 6, réalisme thermique accru

    # Machines
    nb_machines = 30
    machine_ids = [f"GDIZ_TEX_M_{i:02d}" for i in range(1, nb_machines + 1)]
    machine_type_quality = ["L", "M", "H"]

    # Usure — calibré pour ~4-8 pannes / machine / 2 ans (moyenne empirique : 6.6).
    # Raisonnement : 0.0022 (base 5 min) × 6 = 0.013 théorique, mais les facteurs
    # multiplicatifs poussière/tension (Harmattan : dust_factor ≈ 1.6-2.0,
    # voltage_factor ≈ 1.3) portent l'incrément effectif moyen à ~0.008-0.010/pas.
    # La valeur de base doit donc rester à 0.005 pour rester dans la plage cible.
    base_wear_increment = 0.005
    initial_wear_range = (10, 70)
    critical_wear_threshold = 85

    # Thermique — inertie machine forte : 30 min ≈ 60 % du chemin vers T_cible
    coef_lissage_temp = 0.55
    friction_temp_factor = 25.0

    # Électrique et environnement
    nominal_voltage = 230.0
    voltage_variation_range = 0.15

    # Erreurs capteurs IoT
    missing_data_ratio = 0.02

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
        """Détermine la saison selon le calendrier du Sud-Bénin."""
        months = timestamps.month.values
        seasons = np.ones(len(timestamps), dtype=int)
        rainy_months = [4, 5, 6, 7, 10]
        seasons[np.isin(months, rainy_months)] = 0
        return seasons

    def generate_ambient_temperature(self, timestamps: pd.DatetimeIndex, seasons: np.ndarray) -> np.ndarray:
        """
        Température de l'atelier combinant normales saisonnières et cycle jour/nuit.
        Régulation industrielle : si T_brute > 38°C, la ventilation dissipe 70 % de l'excès.
        """
        hours = timestamps.hour.values

        daily_variation = 4.5 * np.sin(2 * np.pi * (hours - 8) / 24)
        base_temp = np.where(seasons == 1, 33.5, 29.5)
        noise = np.where(seasons == 1, np.random.normal(0, 1.0, len(timestamps)), np.random.normal(0, 1.8, len(timestamps)))

        ambient_temp = base_temp + daily_variation + noise

        # Régulation/ventilation industrielle : dissipe 70 % de la chaleur excédentaire > 38°C
        ambient_temp = np.where(ambient_temp > 38.0, 38.0 + 0.30 * (ambient_temp - 38.0), ambient_temp)
        return np.clip(ambient_temp, 23.0, 42.0)

    def generate_rain_events(self, seasons: np.ndarray) -> np.ndarray:
        """
        Génère des épisodes de pluie cohérents via chaîne de Markov.
        Probabilités recalibrées pour un pas de 30 min (≈ 6× les valeurs 5 min).
        """
        num_samples = len(seasons)
        rain_flag = np.zeros(num_samples, dtype=int)

        for i in range(1, num_samples):
            if seasons[i] == 0:  # Saison des pluies
                p_start, p_stop = 0.12, 0.40
            else:  # Saison sèche / Harmattan
                p_start, p_stop = 0.006, 0.75

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

        shape = np.select(condlist=[is_harmattan, seasons == 1], choicelist=[5.0, 2.5], default=1.5)
        scale = np.select(condlist=[is_harmattan, seasons == 1], choicelist=[30.0, 12.0], default=6.0)

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
        Les coupures durent en moyenne ~50 min (continuation = 0.40).
        """
        hours = timestamps.hour.values
        num_samples = len(timestamps)

        base_prob = 0.0006  # ≈ 6× la valeur 5 min (0.0001)
        peak_hours = np.isin(hours, [11, 12, 13, 14, 19, 20, 21, 22])
        failure_probability = np.where(peak_hours, base_prob * 8, base_prob)

        power_loss = (np.random.random(num_samples) < failure_probability).astype(int)

        # Persistance réduite : avec des pas de 30 min, E[durée] ≈ 1.7 pas ≈ 50 min
        for i in range(1, num_samples):
            if power_loss[i - 1] == 1 and np.random.random() < 0.40:
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

    def __init__(self, machine_id: str, machine_type: str, df_global: pd.DataFrame, config: SyntheticDataConfig = None, override_params: dict = None):
        self.machine_id = machine_id
        self.machine_type = machine_type
        self.df_global = df_global
        self.config = config or SyntheticDataConfig()

        # RNG isolé par machine — évite toute pollution croisée des flux aléatoires
        machine_seed = self.config.global_seed + hash(machine_id) % 10000
        self.rng = np.random.default_rng(machine_seed)

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

        self.tool_wear[0] = self.rng.uniform(*self.config.initial_wear_range)
        self.process_temperature[0] = self.ambient_array[0] + 5.0

    def _update_physics(self, idx: int, restarted_from_blackout: bool = False):
        """Calcule les indicateurs physiques de la machine lorsqu'elle est en fonctionnement."""
        activity_idx = self.activity_level_array[idx]
        stability_pct = self.stability_array[idx] / 100.0
        voltage_deviation = max(0, self.config.nominal_voltage - self.voltage_array[idx])

        # 1. Vitesse & Couple
        base_speed = 1500.0
        self.rotational_speed[idx] = base_speed * activity_idx * (0.9 + 0.1 * stability_pct)
        self.rotational_speed[idx] += self.rng.normal(0, 25)
        self.rotational_speed[idx] = max(0, self.rotational_speed[idx])

        base_torque = 45.0
        self.torque[idx] = base_torque * activity_idx * (1.3 - 0.3 * stability_pct)
        self.torque[idx] += self.rng.normal(0, 3)
        self.torque[idx] = max(0, self.torque[idx])

        # 2. Usure cumulative
        wear_prev = self.tool_wear[idx - 1]
        if restarted_from_blackout:
            wear_prev = min(wear_prev + self.params["power_loss_restart_malus"], 100)

        dust_factor = 1.0 + 0.004 * self.dust_array[idx] * self.params["dust_sensitivity"]
        voltage_factor = 1.0 + 0.6 * (1.0 - stability_pct) * self.params["voltage_sensitivity"]
        wear_increment = self.config.base_wear_increment * self.params["wear_multiplier"]
        wear_increment *= dust_factor * voltage_factor
        self.tool_wear[idx] = min(wear_prev + wear_increment, 100)

        # 3. Température Machine
        temp_prev = self.process_temperature[idx - 1]
        electrical_heat = (voltage_deviation * 0.15) * self.params["voltage_sensitivity"]
        target_temp = (
            self.ambient_array[idx]
            + (self.config.friction_temp_factor * (self.torque[idx] / 35.0))
            + self.params["temp_target_offset"]
            + electrical_heat
        )
        alpha = self.config.coef_lissage_temp
        self.process_temperature[idx] = alpha * target_temp + (1 - alpha) * temp_prev

        # 4. Vibrations
        wear_norm = self.tool_wear[idx] / 100.0
        base_vibration = 0.4 * (self.rotational_speed[idx] / base_speed) + 0.4 * (1.0 - stability_pct)
        if restarted_from_blackout:
            base_vibration += self.rng.uniform(3.0, 5.0) * self.params["wear_multiplier"]
        wear_effect = 4.5 * (wear_norm**2)
        noise = self.rng.normal(0, self.params["vibration_noise"])
        self.vibration[idx] = np.clip(base_vibration + wear_effect + noise, 0.05, 15.0)

    def _calculate_failure_probability(self, idx: int) -> float:
        """Évalue la probabilité de panne au pas actuel."""
        wear = self.tool_wear[idx]
        temp = self.process_temperature[idx]
        ambient = self.ambient_array[idx]
        stability_pct = self.stability_array[idx] / 100.0
        vibration = self.vibration[idx]

        prob = 0.000005  # Risque résiduel de base

        # 1a. Panne mécanique : zone pré-critique L (80 % - 85 %)
        #     Les machines L cassent souvent avant d'atteindre le seuil universel.
        if self.machine_type == "L" and 80.0 < wear <= self.config.critical_wear_threshold:
            prob += 0.04 * np.exp((wear - 80.0) / 5.0)

        # 1b. Panne mécanique : seuil critique universel (85 %)
        if wear > self.config.critical_wear_threshold:
            prob += 0.12 * np.exp((wear - self.config.critical_wear_threshold) / 5.0)

        # 1c. Panne par vibration excessive — spécifique L
        #     Vibration > 3,5 mm/s indique un jeu mécanique dangereux (usure ≥ 80 %).
        if self.machine_type == "L" and vibration > 3.5:
            prob += 0.018 * (vibration - 3.5)

        # 2. Panne thermique
        if temp > ambient + 25.0:
            prob += 0.000008 * np.exp((temp - ambient - 25.0) / 4.0)

        # 3. Panne électrique par instabilité routinière du réseau
        elec_sensitivity = self.params.get("ups_elec_sensitivity", self.params["voltage_sensitivity"])
        if stability_pct < 0.85:
            prob += 0.00005 * (1.0 - stability_pct) * elec_sensitivity

        return min(prob, 1.0)

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

            # Refroidissement passif accéléré : en 30 min, la machine retombe quasi à l'ambiant
            self.process_temperature[idx] = (
                alpha_refroidissement * self.ambient_array[idx] + (1 - alpha_refroidissement) * self.process_temperature[idx - 1]
            )

        # Remise à neuf de l'outil au dernier pas de maintenance
        if end_downtime < self.num_samples:
            self.tool_wear[end_downtime - 1] = 0.0

        return end_downtime

    def evolve_states(self) -> pd.DataFrame:
        """Exécute la boucle d'évolution séquentielle sur la chronologie de la machine."""
        maintenance_end_idx = 0

        for idx in range(1, self.num_samples):
            if idx < maintenance_end_idx:
                continue

            # B. Interception des délestages et coupures de courant
            if self.power_loss_array[idx] == 1:
                self.rotational_speed[idx] = 0.0
                self.torque[idx] = 0.0
                self.vibration[idx] = np.abs(self.rng.normal(0, 0.01))
                self.tool_wear[idx] = self.tool_wear[idx - 1]
                self.failure[idx] = 0

                alpha = self.config.coef_lissage_temp
                self.process_temperature[idx] = alpha * self.ambient_array[idx] + (1 - alpha) * self.process_temperature[idx - 1]

                if self.power_loss_array[idx - 1] == 0:
                    # L=4%, M=3.2%, H=0.8% — H protégé par l'UPS au choc de coupure
                    electric_shock_prob = 0.04 * self.params["blackout_failure_sensitivity"]
                    if self.rng.random() < electric_shock_prob:
                        self.failure[idx] = 1
                        maintenance_end_idx = self._handle_downtime_phase(idx)
                continue

            restarted_from_blackout = self.power_loss_array[idx - 1] == 1

            # C. Arrêts planifiés (week-end, activité nulle)
            if self.activity_level_array[idx] == 0.0:
                self.rotational_speed[idx] = 0.0
                self.torque[idx] = 0.0
                self.vibration[idx] = np.abs(self.rng.normal(0, 0.01))
                self.tool_wear[idx] = self.tool_wear[idx - 1]
                self.failure[idx] = 0

                alpha = self.config.coef_lissage_temp
                self.process_temperature[idx] = alpha * self.ambient_array[idx] + (1 - alpha) * self.process_temperature[idx - 1]
                continue

            # D. Régime de fonctionnement normal
            self._update_physics(idx, restarted_from_blackout=restarted_from_blackout)

            failure_prob = self._calculate_failure_probability(idx)
            if self.rng.random() < failure_prob:
                self.failure[idx] = 1
                maintenance_end_idx = self._handle_downtime_phase(idx)

        return pd.DataFrame(
            {
                "machine_id": self.machine_id,
                "machine_type": self.machine_type,
                "timestamp": self.df_global["timestamp"],
                "rotational_speed": self.rotational_speed,
                "torque": self.torque,
                "tool_wear": self.tool_wear,
                "process_temperature": self.process_temperature,
                "vibration": self.vibration,
                "target": self.failure,
            }
        )


# C - Agrégation, masquage IoT et persistence des données


class DataAggregator:
    """Combine les historiques des machines et orchestre la persistence des données."""

    def __init__(self, config: SyntheticDataConfig = None):
        self.config = config or SyntheticDataConfig()

    def inject_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Simule des pertes de paquets IoT ou des défaillances de capteurs physiques.
        Utilise une sous-matrice float explicite pour éviter les comportements instables
        lors de l'affectation de NaN sur un DataFrame à types mixtes.
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

        ratio = self.config.missing_data_ratio
        total_sensor_cells = len(df_copy) * len(sensor_cols)
        num_missing = int(total_sensor_cells * ratio)

        if num_missing == 0:
            return df_copy

        random_rows = np.random.randint(0, len(df_copy), size=num_missing)
        random_sub_cols = np.random.randint(0, len(sensor_cols), size=num_missing)

        # Extraction explicite en float → injection NaN → réassignation propre
        sensor_data = df_copy[sensor_cols].to_numpy(dtype=float)
        sensor_data[random_rows, random_sub_cols] = np.nan
        df_copy[sensor_cols] = sensor_data

        logger.info(f"Défaillance capteurs : {num_missing} valeurs manquantes injectées.")
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

    def __init__(self, random_seed: int = 24):
        self.config = SyntheticDataConfig()
        self.config.global_seed = random_seed
        np.random.seed(random_seed)

    def generate(self) -> pd.DataFrame:
        exo_gen = ExogenousContextGenerator(self.config)
        df_global = exo_gen.generate_global_context()

        # Distribution déterministe : exactement 10 machines de chaque type (L, M, H)
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
