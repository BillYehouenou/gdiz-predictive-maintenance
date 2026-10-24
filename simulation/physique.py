"""
Physique de l'atelier et des machines : la météo réelle devient des conditions intérieures, et l'état caché
de chaque machine (usure, encrassement, roulements) pilote des risques de panne concurrents (AI4I 2020).
Seuls les capteurs et les compteurs GMAO sont exportés.
"""

import hashlib
from datetime import datetime

import numpy as np
import pandas as pd

from simulation.meteo import au_pas
from src.config import CHEMIN_DB


class ConfigDonneesSynthetiques:
    graine_globale = 42
    date_debut = datetime(2024, 1, 1)
    duree_jours = 730  # 2 cycles Harmattan
    intervalle_minutes = 30
    nb_machines = 50
    ids_machines = [f"GDIZ_TEX_M_{i:02d}" for i in range(1, nb_machines + 1)]
    fichier_duckdb = CHEMIN_DB

    # Usure outil (%) : ~4 à 8 pannes par machine sur 2 ans
    increment_usure_base = 0.005
    plage_usure_initiale = (10, 70)

    # TWF : seuil de rupture propre à chaque machine, dans [65, 99] %, risque en rampe juste avant
    seuil_min_use = 65.0
    plage_seuil_use = 34.0
    beta_a_seuil_use = 4.0
    beta_b_seuil_use = 2.0
    bande_rampe_use = 12.0
    exposant_rampe_use = 3.0
    coeff_risque_use = 0.08

    # HDF : surchauffe soutenue sur 4h, au-delà de 38 °C d'écart à l'ambiant
    pas_fenetre_shd = 8
    seuil_surchauffe_shd = 38.0
    coeff_risque_shd = 0.00012

    # OSF : couple élevé sur un outil ou des roulements usés
    coeff_risque_sm = 0.0007
    taux_base_usure_roulements = 0.004
    coeff_vibration_roulements = 2.0
    poids_roulements_osf = 0.25

    probabilite_choc_coupure = 0.04  # PWF : choc au retour du courant
    risque_base_pa = 0.000018  # RNF : risque résiduel sans précurseur

    # Chaleur et encrassement (poussière --> moins de dissipation --> HDF)
    facteur_chaleur_usure = 3.0  # °C à usure 100 %
    taux_base_encrassement = 0.06  # % par pas à 100 µg/m³
    facteur_chaleur_encrassement = 1.8  # °C à encrassement 100 %
    residu_reinitialisation_encrassement = 10.0
    coef_lissage_temperature = 0.55
    facteur_temperature_frottement = 25.0

    # Maintenance préventive : visite aléatoire, plus probable sur une machine usée
    probabilite_base_preventive = 0.00002
    coeff_risque_preventif = 0.0001
    heures_mediane_duree_preventive = 2.0
    heures_max_duree_preventive = 8.0
    reduction_roulements_preventive = 0.5

    # Atelier : extérieur --> intérieur
    demi_vie_inertie_thermique_pas = 6  # ~3h
    surchauffe_atelier_base = 4.0  # °C (toiture tôle, éclairage)
    surchauffe_atelier_activite = 2.5  # °C à pleine production
    seuil_ventilation = 38.0  # °C, au-delà la ventilation dissipe 70 % de l'excès
    coeff_humidite_chauffage = 0.065  # -6,5 % d'humidité relative par °C de réchauffement
    demi_vie_infiltration_poussiere_pas = 4  # ~2h
    taux_infiltration_poussiere = 0.7
    poussiere_fibres_base = 12.0  # µg/m³ de fibres textiles, machines à l'arrêt
    poussiere_fibres_activite = 25.0  # µg/m³ en plus à pleine production

    # Réseau SBEE
    tension_nominale = 220.0
    probabilite_coupure = 0.0006  # par pas, x8 aux heures de pointe
    multiplicateur_pointe_coupure = 8
    persistance_coupure = 0.55  # durée moyenne ~1h

    # Défaillances capteurs : rafales de NaN, capteurs figés, dérives de calibration
    probabilite_debut_rafale_nan_capteur = 0.0024
    pas_moyens_duree_rafale_nan_capteur = 3.0
    sigma_duree_rafale_nan_capteur = 0.6
    plage_duree_rafale_nan_capteur = (1, 24)
    probabilite_debut_capteur_bloque = 0.00025
    pas_moyens_duree_capteur_bloque = 12.0
    sigma_duree_capteur_bloque = 1.0
    plage_duree_capteur_bloque = (2, 96)
    probabilite_debut_derive_capteur = 0.00006
    pas_moyens_duree_derive_capteur = 48.0
    sigma_duree_derive_capteur = 1.0
    plage_duree_derive_capteur = (12, 200)
    pct_max_derive_capteur = 0.5  # biais final, en écart-types du capteur


# A - Contexte exogène global


class GenerateurContexteExogene:
    """Génère la chronologie environnementale et électrique partagée par toute l'usine."""

    def __init__(self, config: ConfigDonneesSynthetiques | None = None):
        self.config = config or ConfigDonneesSynthetiques()
        self.rng = np.random.default_rng(self.config.graine_globale)

    def generer_horodatages(self) -> pd.DatetimeIndex:
        nb_pas = int(self.config.duree_jours * 24 * 60 / self.config.intervalle_minutes)
        return pd.date_range(start=self.config.date_debut, periods=nb_pas, freq=f"{self.config.intervalle_minutes}min")

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

        bruit_operationnel = self.rng.normal(0, 0.04, len(timestamps))
        activite_locale = activite_locale + bruit_operationnel
        activite_locale = np.where(jours >= 5, 0.0, activite_locale)
        return np.clip(activite_locale, 0.0, 1.0)

    def generer_temperature_ambiante(self, temperature_exterieure: np.ndarray, niveau_activite: np.ndarray) -> np.ndarray:
        """
        Température de l'atelier : extérieur lissé par l'inertie du bâtiment + chaleur des machines.
        Au-delà du seuil de ventilation, 70 % de l'excès est dissipé.
        """
        c = self.config
        inertie = pd.Series(temperature_exterieure).ewm(halflife=c.demi_vie_inertie_thermique_pas).mean().to_numpy()
        temperature = inertie + c.surchauffe_atelier_base + c.surchauffe_atelier_activite * niveau_activite
        temperature += self.rng.normal(0, 0.3, len(temperature))
        return np.where(
            temperature > c.seuil_ventilation, c.seuil_ventilation + 0.30 * (temperature - c.seuil_ventilation), temperature
        )

    def generer_humidite(
        self, humidite_exterieure: np.ndarray, temperature_exterieure: np.ndarray, temperature_ambiante: np.ndarray
    ) -> np.ndarray:
        """Air extérieur réchauffé dans l'atelier à humidité absolue constante : l'humidité relative baisse."""
        rechauffement = temperature_ambiante - temperature_exterieure
        humidite = humidite_exterieure * np.exp(-self.config.coeff_humidite_chauffage * rechauffement)
        return np.clip(humidite + self.rng.normal(0, 1.5, len(humidite)), 20, 98)

    def generer_concentration_poussiere(self, poussiere_exterieure: np.ndarray, niveau_activite: np.ndarray) -> np.ndarray:
        """Poussière intérieure : infiltration de la poussière désertique (Harmattan) + fibres textiles de la production."""
        c = self.config
        infiltration = pd.Series(poussiere_exterieure).ewm(halflife=c.demi_vie_infiltration_poussiere_pas).mean().to_numpy()
        fibres = c.poussiere_fibres_base + c.poussiere_fibres_activite * niveau_activite
        bruit = self.rng.lognormal(0.0, 0.15, len(fibres))
        return (fibres + c.taux_infiltration_poussiere * infiltration) * bruit

    def generer_niveau_tension(self, niveau_activite: np.ndarray) -> np.ndarray:
        tension_base = self.config.tension_nominale
        variation_charge = -15 * niveau_activite
        bruit = self.rng.normal(0, 2.5, len(niveau_activite))
        return tension_base + variation_charge + bruit

    def generer_indicateur_coupure(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """
        Simule des délestages SBEE, plus fréquents aux heures de pointe, de durée géométrique.
        Réseau volontairement clément (hypothèse : alimentation dédiée GDIZ).
        """
        c = self.config
        heures_pointe = np.isin(timestamps.hour.values, [11, 12, 13, 14, 19, 20, 21, 22])
        probabilite = np.where(heures_pointe, c.probabilite_coupure * c.multiplicateur_pointe_coupure, c.probabilite_coupure)

        coupure = np.zeros(len(timestamps), dtype=int)
        debuts = np.flatnonzero(self.rng.random(len(timestamps)) < probabilite)
        durees = self.rng.geometric(1 - c.persistance_coupure, len(debuts))
        for debut, duree in zip(debuts, durees):
            coupure[debut : debut + duree] = 1
        return coupure

    def generer_contexte_global(self, meteo_horaire: pd.DataFrame) -> pd.DataFrame:
        timestamps = self.generer_horodatages()
        meteo = au_pas(meteo_horaire, timestamps)
        temperature_exterieure = meteo["temperature_exterieure"].to_numpy()

        niveau_activite = self.generer_niveau_activite(timestamps)
        temperature_ambiante = self.generer_temperature_ambiante(temperature_exterieure, niveau_activite)

        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "niveau_activite": niveau_activite,
                "temperature_ambiante": temperature_ambiante,
                "humidite": self.generer_humidite(
                    meteo["humidite_exterieure"].to_numpy(), temperature_exterieure, temperature_ambiante
                ),
                "precipitation_mm": meteo["precipitation_mm"].to_numpy(),
                "concentration_poussiere": self.generer_concentration_poussiere(
                    meteo["poussiere_exterieure"].to_numpy(), niveau_activite
                ),
                "niveau_tension": self.generer_niveau_tension(niveau_activite),
                "indicateur_coupure": self.generer_indicateur_coupure(timestamps),
            }
        )


# B - Moteur d'évolution chronologique par machine


class EvolueurEtatMachine:
    """Gère l'évolution d'état chronologique d'une machine (Dépendance physique N-1)."""

    def __init__(
        self,
        id_machine: str,
        type_machine: str,
        df_global: pd.DataFrame,
        config: ConfigDonneesSynthetiques | None = None,
        parametres_surcharge: dict | None = None,
    ):
        self.id_machine = id_machine
        self.type_machine = type_machine
        self.df_global = df_global
        self.config = config or ConfigDonneesSynthetiques()
        self.pas_heures = self.config.intervalle_minutes / 60.0

        # RNG isolé par machine pour éviter toute pollution croisée des flux aléatoires
        hash_machine = int(hashlib.md5(id_machine.encode("utf-8")).hexdigest(), 16) % 10000
        self.rng = np.random.default_rng(self.config.graine_globale + hash_machine)

        # Seuil de rupture personnel (TWF)
        self.seuil_use = self.config.seuil_min_use + self.config.plage_seuil_use * self.rng.beta(
            self.config.beta_a_seuil_use, self.config.beta_b_seuil_use
        )

        self.nb_echantillons = len(df_global)
        self.tableau_temperature_ambiante = df_global["temperature_ambiante"].values
        self.tableau_poussiere = df_global["concentration_poussiere"].values
        self.tableau_tension = df_global["niveau_tension"].values
        # Stabilité réseau (0-1) : grandeur physique interne, dérivée de l'écart à la tension nominale
        self.tableau_stabilite = np.exp(-np.abs(self.tableau_tension - self.config.tension_nominale) / 18)
        self.tableau_niveau_activite = df_global["niveau_activite"].values
        self.tableau_coupures = df_global["indicateur_coupure"].values
        self._definir_parametres_machine()
        if parametres_surcharge:
            self.params.update(parametres_surcharge)
        self._initialiser_tableaux_etat()

    def _definir_parametres_machine(self):
        parametres_type = {
            "L": {  # Entrée de gamme / Rustique : aucune protection électronique
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
                "sensibilite_tension": 2.0,
                "malus_redemarrage_coupure": 0.2,
                # Coupure réseau : l'UPS amortit la surtension -> panne sur blackout rare
                "sensibilite_panne_coupure": 0.2,
            },
        }
        self.params = parametres_type[self.type_machine]

    def _initialiser_tableaux_etat(self):
        n = self.nb_echantillons
        self.vitesse_rotation = np.zeros(n)
        self.couple = np.zeros(n)
        self.temperature_process = np.zeros(n)
        self.vibration = np.zeros(n)
        self.failure = np.zeros(n, dtype=int)
        self.mode_panne = np.full(n, "", dtype=object)
        # marche, arret (planifié), delestage, panne (réparation), entretien (visite préventive)
        self.etat = np.full(n, "marche", dtype=object)

        # État caché : jamais exporté
        self.usure_outil = np.zeros(n)
        self.encrassement = np.zeros(n)
        self.usure_roulements = np.zeros(n)

        # Compteurs GMAO : ce que l'usine connaît réellement de l'état de la machine
        self.heures_outil = np.zeros(n)  # heures de fonctionnement depuis le dernier changement d'outil
        self.heures_maintenance = np.zeros(n)  # heures calendaires depuis la dernière intervention

        self.usure_outil[0] = self.rng.uniform(*self.config.plage_usure_initiale)
        # Heures d'outil cohérentes avec l'usure initiale (au rythme d'usure nominal du type)
        usure_par_heure = self.config.increment_usure_base * self.params["multiplicateur_usure"] / self.pas_heures
        self.heures_outil[0] = self.usure_outil[0] / usure_par_heure
        self.heures_maintenance[0] = self.rng.uniform(0, 720)
        self.temperature_process[0] = self.tableau_temperature_ambiante[0] + 5.0

    def _arreter(self, idx: int, alpha_refroidissement: float, etat: str):
        """Machine à l'arrêt : pas de rotation, état caché figé, refroidissement vers l'ambiant."""
        self.etat[idx] = etat
        self.vitesse_rotation[idx] = 0.0
        self.couple[idx] = 0.0
        self.vibration[idx] = np.abs(self.rng.normal(0, 0.01))
        for serie in (self.usure_outil, self.encrassement, self.usure_roulements, self.heures_outil):
            serie[idx] = serie[idx - 1]
        self.heures_maintenance[idx] = self.heures_maintenance[idx - 1] + self.pas_heures
        self.temperature_process[idx] = (
            alpha_refroidissement * self.tableau_temperature_ambiante[idx]
            + (1 - alpha_refroidissement) * self.temperature_process[idx - 1]
        )

    def _mettre_a_jour_physique(self, idx: int, redemarre_apres_coupure: bool = False):
        """Calcule les indicateurs physiques de la machine lorsqu'elle est en fonctionnement."""
        indice_activite = self.tableau_niveau_activite[idx]
        pourcentage_stabilite = self.tableau_stabilite[idx]
        ecart_tension = max(0, self.config.tension_nominale - self.tableau_tension[idx])

        self.heures_outil[idx] = self.heures_outil[idx - 1] + self.pas_heures
        self.heures_maintenance[idx] = self.heures_maintenance[idx - 1] + self.pas_heures

        # 1. Vitesse & Couple
        vitesse_base = 1500.0
        self.vitesse_rotation[idx] = vitesse_base * (0.97 + 0.03 * pourcentage_stabilite)
        self.vitesse_rotation[idx] += self.rng.normal(0, 12)
        self.vitesse_rotation[idx] = max(0, self.vitesse_rotation[idx])

        couple_base = 45.0
        self.couple[idx] = couple_base * indice_activite * (1.3 - 0.3 * pourcentage_stabilite)
        self.couple[idx] += self.rng.normal(0, 3)
        self.couple[idx] = max(0, self.couple[idx])

        # 2a. Usure cumulative, non plafonnée : au-delà du seuil personnel, le risque TWF continue de croître
        usure_precedente = self.usure_outil[idx - 1]
        if redemarre_apres_coupure:
            usure_precedente += self.params["malus_redemarrage_coupure"]

        dust_factor = 1.0 + 0.004 * self.tableau_poussiere[idx] * self.params["sensibilite_poussiere"]
        voltage_factor = 1.0 + 0.6 * (1.0 - pourcentage_stabilite) * self.params["sensibilite_tension"]
        increment_usure = self.config.increment_usure_base * self.params["multiplicateur_usure"]
        increment_usure *= dust_factor * voltage_factor
        self.usure_outil[idx] = usure_precedente + increment_usure

        # 2b. Encrassement
        increment_encrassement = (
            self.config.taux_base_encrassement * (self.tableau_poussiere[idx] / 100.0) * self.params["sensibilite_poussiere"]
        )
        self.encrassement[idx] = min(self.encrassement[idx - 1] + increment_encrassement, 100.0)

        # 2c. Usure des roulements
        ratio_vitesse = self.vitesse_rotation[idx] / vitesse_base
        increment_usure_roulements = (
            self.config.taux_base_usure_roulements * (ratio_vitesse**2) * self.params["multiplicateur_usure"]
        )
        self.usure_roulements[idx] = min(self.usure_roulements[idx - 1] + increment_usure_roulements, 100.0)

        # 3. Température Machine
        temperature_precedente = self.temperature_process[idx - 1]
        chaleur_electrique = (ecart_tension * 0.15) * self.params["sensibilite_tension"]
        chaleur_usure = self.config.facteur_chaleur_usure * (self.usure_outil[idx] / 100.0)
        # Encrassement : dissipation thermique réduite par la poussière accumulée.
        chaleur_encrassement = self.config.facteur_chaleur_encrassement * (self.encrassement[idx] / 100.0)
        temperature_cible = (
            self.tableau_temperature_ambiante[idx]
            + (self.config.facteur_temperature_frottement * (self.couple[idx] / 35.0))
            + self.params["decalage_temperature_cible"]
            + chaleur_electrique
            + chaleur_usure
            + chaleur_encrassement
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
        bruit = self.rng.normal(0, self.params["bruit_vibration"])
        self.vibration[idx] = np.clip(vibration_base + effet_usure + effet_roulements + bruit, 0.05, 15.0)

    def _calculer_risques_panne(self, idx: int) -> dict:
        """
        Évalue le hasard de panne par cause au pas actuel (modèle à risques concurrents,
        façon AI4I 2020). Chaque cause a sa propre signature précurseur ; aucune ne repose
        sur un seuil dur déterministe. PWF est tiré à part, au retour du courant.
        """
        wear = self.usure_outil[idx]
        usure_normalisee = wear / 100.0
        couple = self.couple[idx]

        # TWF : rampe lisse autour du seuil de rupture personnel de la machine
        band = self.config.bande_rampe_use
        excess = max(0.0, (wear - (self.seuil_use - band)) / band)
        twf = self.config.coeff_risque_use * (excess**self.config.exposant_rampe_use) * self.params["multiplicateur_usure"]

        # HDF : surchauffe soutenue sur une fenêtre glissante de ~4h
        window = min(idx, self.config.pas_fenetre_shd)
        overheat = self.temperature_process[idx - window : idx + 1] - self.tableau_temperature_ambiante[idx - window : idx + 1]
        surchauffe_soutenue = max(0.0, float(overheat.mean()) - self.config.seuil_surchauffe_shd)
        hdf = self.config.coeff_risque_shd * surchauffe_soutenue**2

        # OSF : surcharge mécanique couple élevé combiné à une usure déjà significative
        usure_roulements_normalisee = self.usure_roulements[idx] / 100.0
        terme_usure_osf = (
            1.0 - self.config.poids_roulements_osf
        ) * usure_normalisee**3 + self.config.poids_roulements_osf * usure_roulements_normalisee**3
        osf = self.config.coeff_risque_sm * (couple / 45.0) * terme_usure_osf

        return {"TWF": twf, "HDF": hdf, "OSF": osf, "RNF": self.config.risque_base_pa}

    def _gerer_phase_arret(self, index_panne: int) -> int:
        """
        Réparation suite à une panne : durée log-normale (médiane ≈ 3h, queue jusqu'à 48h),
        refroidissement passif accéléré. Le composant en cause est remplacé.
        """
        duree_heures = float(np.clip(self.rng.lognormal(mean=np.log(3.0), sigma=1.2), 0.5, 48.0))
        fin_arret = min(index_panne + 1 + int(np.ceil(duree_heures / self.pas_heures)), self.nb_echantillons)
        self.etat[index_panne] = "panne"
        for idx in range(index_panne + 1, fin_arret):
            self._arreter(idx, alpha_refroidissement=0.70, etat="panne")

        if fin_arret < self.nb_echantillons:
            dernier = fin_arret - 1
            cause = self.mode_panne[index_panne]
            if cause == "TWF":
                self.usure_outil[dernier] = 0.0
                self.heures_outil[dernier] = 0.0
            if cause == "OSF":
                self.usure_roulements[dernier] = 0.0
            self.encrassement[dernier] = min(self.encrassement[dernier], self.config.residu_reinitialisation_encrassement)
            self.heures_maintenance[dernier] = 0.0
        return fin_arret

    def _probabilite_maintenance_preventive(self, idx: int) -> float:
        """Probabilité qu'une visite de maintenance préventive intercepte la machine à ce pas."""
        score_risque = max(self.usure_outil[idx], self.encrassement[idx], self.usure_roulements[idx]) / 100.0
        return self.config.probabilite_base_preventive + self.config.coeff_risque_preventif * score_risque**2

    def _gerer_phase_maintenance_preventive(self, event_idx: int) -> int:
        """
        Visite préventive : pas une panne, plus courte qu'une réparation. L'outil est changé
        et la machine nettoyée, mais les roulements ne sont que révisés.
        """
        c = self.config
        duree_heures = float(
            np.clip(
                self.rng.lognormal(mean=np.log(c.heures_mediane_duree_preventive), sigma=0.6), 0.5, c.heures_max_duree_preventive
            )
        )
        fin = min(event_idx + 1 + int(np.ceil(duree_heures / self.pas_heures)), self.nb_echantillons)
        for idx in range(event_idx + 1, fin):
            self._arreter(idx, alpha_refroidissement=0.70, etat="entretien")

        if fin < self.nb_echantillons:
            dernier = fin - 1
            self.usure_outil[dernier] = 0.0
            self.heures_outil[dernier] = 0.0
            self.usure_roulements[dernier] *= 1.0 - c.reduction_roulements_preventive
            self.encrassement[dernier] = min(self.encrassement[dernier], c.residu_reinitialisation_encrassement)
            self.heures_maintenance[dernier] = 0.0
        return fin

    def evoluer_etats(self) -> pd.DataFrame:
        """Exécute la boucle d'évolution séquentielle sur la chronologie de la machine."""
        index_fin_maintenance = 0
        alpha = self.config.coef_lissage_temperature

        for idx in range(1, self.nb_echantillons):
            if idx < index_fin_maintenance:
                continue

            # Délestage : machine à l'arrêt, choc électrique possible au début de la coupure
            if self.tableau_coupures[idx] == 1:
                self._arreter(idx, alpha, etat="delestage")
                if self.tableau_coupures[idx - 1] == 0:
                    # L=4%, M=3.2%, H=0.8% : H protégé par l'UPS au choc de coupure
                    if self.rng.random() < self.config.probabilite_choc_coupure * self.params["sensibilite_panne_coupure"]:
                        self.failure[idx] = 1
                        self.mode_panne[idx] = "PWF"
                        index_fin_maintenance = self._gerer_phase_arret(idx)
                continue

            # Arrêts planifiés : week-end, activité nulle
            if self.tableau_niveau_activite[idx] == 0.0:
                self._arreter(idx, alpha, etat="arret")
                continue

            self._mettre_a_jour_physique(idx, redemarre_apres_coupure=self.tableau_coupures[idx - 1] == 1)

            hazards = self._calculer_risques_panne(idx)
            total_hazard = sum(hazards.values())
            if self.rng.random() < min(total_hazard, 1.0):
                self.failure[idx] = 1
                weights = np.array(list(hazards.values()))
                self.mode_panne[idx] = self.rng.choice(list(hazards.keys()), p=weights / total_hazard)
                index_fin_maintenance = self._gerer_phase_arret(idx)
            elif self.rng.random() < self._probabilite_maintenance_preventive(idx):
                index_fin_maintenance = self._gerer_phase_maintenance_preventive(idx)

        return pd.DataFrame(
            {
                "id_machine": self.id_machine,
                "type_machine": self.type_machine,
                "timestamp": self.df_global["timestamp"],
                "vitesse_rotation": self.vitesse_rotation,
                "couple": self.couple,
                "temperature_process": self.temperature_process,
                "vibration": self.vibration,
                "heures_depuis_changement_outil": self.heures_outil,
                "heures_depuis_maintenance": self.heures_maintenance,
                "panne": self.failure,
                "mode_panne": self.mode_panne,
                "etat": self.etat,
            }
        )
