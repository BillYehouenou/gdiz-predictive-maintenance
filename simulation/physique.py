"""
Physique de l'atelier et des machines : la météo réelle devient des conditions intérieures, et l'état caché de chaque
machine (usure, encrassement, roulements) pilote des risques de panne concurrents (AI4I 2020). Seuls les capteurs et
les compteurs GMAO sont exportés.
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

    # Usure outil (%) : ~12 à 16 pannes par machine sur 2 ans
    increment_usure_base, plage_usure_initiale = 0.011, (10, 70)
    # TWF : seuil de rupture propre à chaque machine dans [65, 99] %, risque en rampe juste avant
    seuil_min_use, plage_seuil_use, beta_a_seuil_use, beta_b_seuil_use = 65.0, 34.0, 4.0, 2.0
    bande_rampe_use, exposant_rampe_use, coeff_risque_use = 12.0, 3.0, 0.08
    # HDF : surchauffe soutenue sur 4 h au-delà de 38 °C d'écart à l'ambiant
    pas_fenetre_shd, seuil_surchauffe_shd, coeff_risque_shd = 8, 38.0, 0.00035
    # OSF : couple élevé sur un outil ou des roulements usés
    coeff_risque_sm, taux_base_usure_roulements, coeff_vibration_roulements, poids_roulements_osf = 0.0018, 0.004, 2.0, 0.25
    probabilite_choc_coupure = 0.04  # PWF : choc au retour du courant
    risque_base_pa = 0.000018  # RNF : risque résiduel sans précurseur

    # Chaleur et encrassement
    facteur_chaleur_usure, facteur_chaleur_encrassement = 3.0, 1.8  # °C à 100 %
    taux_base_encrassement, residu_reinitialisation_encrassement = 0.06, 10.0  # % par pas à 100 µg/m³
    coef_lissage_temperature, facteur_temperature_frottement = 0.55, 25.0

    # Maintenance préventive : visite aléatoire, plus probable sur une machine usée
    probabilite_base_preventive, coeff_risque_preventif = 0.00002, 0.0001
    heures_mediane_duree_preventive, heures_max_duree_preventive, reduction_roulements_preventive = 2.0, 8.0, 0.5

    # Atelier : extérieur --> intérieur (inertie ~3 h, toiture tôle, ventilation au-delà de 38 °C)
    demi_vie_inertie_thermique_pas, surchauffe_atelier_base, surchauffe_atelier_activite = 6, 4.0, 2.5
    seuil_ventilation, coeff_humidite_chauffage = 38.0, 0.065  # -6,5 % d'humidité relative par °C
    demi_vie_infiltration_poussiere_pas, taux_infiltration_poussiere = 4, 0.7
    poussiere_fibres_base, poussiere_fibres_activite = 12.0, 25.0  # µg/m³ de fibres textiles

    # Réseau SBEE : coupures x8 aux heures de pointe, durée moyenne ~1 h
    tension_nominale, probabilite_coupure, multiplicateur_pointe_coupure, persistance_coupure = 220.0, 0.0006, 8, 0.55

    # Défaillances capteurs : (probabilité de début, durée moyenne en pas, sigma, plage de durée)
    rafale_nan = (0.0024, 3.0, 0.6, (1, 24))
    capteur_bloque = (0.00025, 12.0, 1.0, (2, 96))
    derive_capteur = (0.00006, 48.0, 1.0, (12, 200))
    pct_max_derive_capteur = 0.5  # biais final, en écart-types du capteur


class GenerateurContexteExogene:
    """Chronologie environnementale et électrique partagée par toute l'usine."""

    def __init__(self, config: ConfigDonneesSynthetiques | None = None):
        self.config = config or ConfigDonneesSynthetiques()
        self.rng = np.random.default_rng(self.config.graine_globale)

    def generer_horodatages(self) -> pd.DatetimeIndex:
        nb_pas = int(self.config.duree_jours * 24 * 60 / self.config.intervalle_minutes)
        return pd.date_range(start=self.config.date_debut, periods=nb_pas, freq=f"{self.config.intervalle_minutes}min")

    def generer_niveau_activite(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """Trois équipes en 3x8, relève moins productive, arrêt le week-end."""
        heures, minutes, jours = timestamps.hour.values, timestamps.minute.values, timestamps.dayofweek.values
        activite = np.zeros(len(timestamps))
        activite = np.where((heures >= 6) & (heures < 14), 0.92, activite)
        activite = np.where((heures >= 14) & (heures < 22), 0.86, activite)
        activite = np.where((heures >= 22) | (heures < 6), 0.55, activite)
        activite = np.where((minutes < 20) & np.isin(heures, [6, 14, 22]), activite * 0.75, activite)
        activite = activite + self.rng.normal(0, 0.04, len(timestamps))
        return np.clip(np.where(jours >= 5, 0.0, activite), 0.0, 1.0)

    def generer_temperature_ambiante(self, temperature_exterieure: np.ndarray, niveau_activite: np.ndarray) -> np.ndarray:
        """Extérieur lissé par l'inertie du bâtiment + chaleur des machines ; la ventilation dissipe 70 % de l'excès."""
        c = self.config
        inertie = pd.Series(temperature_exterieure).ewm(halflife=c.demi_vie_inertie_thermique_pas).mean().to_numpy()
        temperature = inertie + c.surchauffe_atelier_base + c.surchauffe_atelier_activite * niveau_activite
        temperature += self.rng.normal(0, 0.3, len(temperature))
        trop_chaud = temperature > c.seuil_ventilation
        return np.where(trop_chaud, c.seuil_ventilation + 0.30 * (temperature - c.seuil_ventilation), temperature)

    def generer_humidite(self, humidite_ext: np.ndarray, temperature_ext: np.ndarray, temperature_amb: np.ndarray) -> np.ndarray:
        """Air extérieur réchauffé à humidité absolue constante : l'humidité relative baisse."""
        humidite = humidite_ext * np.exp(-self.config.coeff_humidite_chauffage * (temperature_amb - temperature_ext))
        return np.clip(humidite + self.rng.normal(0, 1.5, len(humidite)), 20, 98)

    def generer_concentration_poussiere(self, poussiere_exterieure: np.ndarray, niveau_activite: np.ndarray) -> np.ndarray:
        """Infiltration de la poussière désertique (Harmattan) + fibres textiles de la production."""
        c = self.config
        infiltration = pd.Series(poussiere_exterieure).ewm(halflife=c.demi_vie_infiltration_poussiere_pas).mean().to_numpy()
        fibres = c.poussiere_fibres_base + c.poussiere_fibres_activite * niveau_activite
        return (fibres + c.taux_infiltration_poussiere * infiltration) * self.rng.lognormal(0.0, 0.15, len(fibres))

    def generer_niveau_tension(self, niveau_activite: np.ndarray) -> np.ndarray:
        return self.config.tension_nominale + -15 * niveau_activite + self.rng.normal(0, 2.5, len(niveau_activite))

    def generer_indicateur_coupure(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """Délestages SBEE de durée géométrique ; réseau clément (hypothèse : alimentation dédiée GDIZ)."""
        c = self.config
        pointe = np.isin(timestamps.hour.values, [11, 12, 13, 14, 19, 20, 21, 22])
        probabilite = np.where(pointe, c.probabilite_coupure * c.multiplicateur_pointe_coupure, c.probabilite_coupure)
        coupure = np.zeros(len(timestamps), dtype=int)
        debuts = np.flatnonzero(self.rng.random(len(timestamps)) < probabilite)
        for debut, duree in zip(debuts, self.rng.geometric(1 - c.persistance_coupure, len(debuts))):
            coupure[debut : debut + duree] = 1
        return coupure

    def generer_contexte_global(self, meteo_horaire: pd.DataFrame) -> pd.DataFrame:
        timestamps = self.generer_horodatages()
        meteo = au_pas(meteo_horaire, timestamps)
        temperature_ext, poussiere_ext = meteo["temperature_exterieure"].to_numpy(), meteo["poussiere_exterieure"].to_numpy()
        activite = self.generer_niveau_activite(timestamps)
        temperature_amb = self.generer_temperature_ambiante(temperature_ext, activite)
        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "niveau_activite": activite,
                "temperature_ambiante": temperature_amb,
                "humidite": self.generer_humidite(meteo["humidite_exterieure"].to_numpy(), temperature_ext, temperature_amb),
                "precipitation_mm": meteo["precipitation_mm"].to_numpy(),
                "concentration_poussiere": self.generer_concentration_poussiere(poussiere_ext, activite),
                "niveau_tension": self.generer_niveau_tension(activite),
                "indicateur_coupure": self.generer_indicateur_coupure(timestamps),
            }
        )


# Par gamme : L rustique sans protection, M standard, H électronique de précision avec onduleur
PARAMETRES_GAMME = {
    "L": dict(multiplicateur_usure=1.3, decalage_temperature_cible=3.0, bruit_vibration=0.25, sensibilite_poussiere=1.5,
              sensibilite_tension=0.5, malus_redemarrage_coupure=0.8, sensibilite_panne_coupure=1.0),
    "M": dict(multiplicateur_usure=1.0, decalage_temperature_cible=0.0, bruit_vibration=0.15, sensibilite_poussiere=1.0,
              sensibilite_tension=1.0, malus_redemarrage_coupure=0.4, sensibilite_panne_coupure=0.8),
    "H": dict(multiplicateur_usure=0.7, decalage_temperature_cible=-2.0, bruit_vibration=0.08, sensibilite_poussiere=0.4,
              sensibilite_tension=2.0, malus_redemarrage_coupure=0.2, sensibilite_panne_coupure=0.2),
}  # fmt: skip


class EvolueurEtatMachine:
    """Évolution pas à pas d'une machine : chaque pas dépend du précédent."""

    def __init__(self, id_machine: str, type_machine: str, df_global: pd.DataFrame,
                 config: ConfigDonneesSynthetiques | None = None, parametres_surcharge: dict | None = None):  # fmt: skip
        self.id_machine, self.type_machine, self.df_global = id_machine, type_machine, df_global
        self.config = c = config or ConfigDonneesSynthetiques()
        self.pas_heures = c.intervalle_minutes / 60.0
        # Un générateur par machine : aucun flux aléatoire partagé entre machines
        self.rng = np.random.default_rng(c.graine_globale + int(hashlib.md5(id_machine.encode()).hexdigest(), 16) % 10000)
        self.seuil_use = c.seuil_min_use + c.plage_seuil_use * self.rng.beta(c.beta_a_seuil_use, c.beta_b_seuil_use)

        self.n = len(df_global)
        self.ambiante = df_global["temperature_ambiante"].values
        self.poussiere = df_global["concentration_poussiere"].values
        self.tension = df_global["niveau_tension"].values
        self.stabilite = np.exp(-np.abs(self.tension - c.tension_nominale) / 18)  # 0-1, écart à la tension nominale
        self.activite = df_global["niveau_activite"].values
        self.coupures = df_global["indicateur_coupure"].values
        self.params = {**PARAMETRES_GAMME[type_machine], **(parametres_surcharge or {})}
        self._initialiser_etat()

    def _initialiser_etat(self):
        n = self.n
        self.vitesse_rotation, self.couple, self.temperature_process, self.vibration = (np.zeros(n) for _ in range(4))
        self.failure = np.zeros(n, dtype=int)
        self.mode_panne = np.full(n, "", dtype=object)
        self.etat = np.full(n, "marche", dtype=object)  # marche, arret, delestage, panne, entretien
        self.usure_outil, self.encrassement, self.usure_roulements = (np.zeros(n) for _ in range(3))  # état caché
        self.heures_outil, self.heures_maintenance = np.zeros(n), np.zeros(n)  # compteurs GMAO

        self.usure_outil[0] = self.rng.uniform(*self.config.plage_usure_initiale)
        usure_par_heure = self.config.increment_usure_base * self.params["multiplicateur_usure"] / self.pas_heures
        self.heures_outil[0] = self.usure_outil[0] / usure_par_heure  # cohérent avec l'usure initiale
        self.heures_maintenance[0] = self.rng.uniform(0, 720)
        self.temperature_process[0] = self.ambiante[0] + 5.0

    def _arreter(self, idx: int, alpha_refroidissement: float, etat: str):
        """Machine à l'arrêt : pas de rotation, état caché figé, refroidissement vers l'ambiant."""
        self.etat[idx] = etat
        self.vitesse_rotation[idx] = self.couple[idx] = 0.0
        self.vibration[idx] = np.abs(self.rng.normal(0, 0.01))
        for serie in (self.usure_outil, self.encrassement, self.usure_roulements, self.heures_outil):
            serie[idx] = serie[idx - 1]
        self.heures_maintenance[idx] = self.heures_maintenance[idx - 1] + self.pas_heures
        a = alpha_refroidissement
        self.temperature_process[idx] = a * self.ambiante[idx] + (1 - a) * self.temperature_process[idx - 1]

    def _mettre_a_jour_physique(self, idx: int, redemarre_apres_coupure: bool = False):
        """Indicateurs physiques d'une machine en fonctionnement."""
        c, p = self.config, self.params
        stabilite = self.stabilite[idx]
        ecart_tension = max(0, c.tension_nominale - self.tension[idx])
        self.heures_outil[idx] = self.heures_outil[idx - 1] + self.pas_heures
        self.heures_maintenance[idx] = self.heures_maintenance[idx - 1] + self.pas_heures

        vitesse_base, couple_base = 1500.0, 45.0
        self.vitesse_rotation[idx] = max(0, vitesse_base * (0.97 + 0.03 * stabilite) + self.rng.normal(0, 12))
        self.couple[idx] = max(0, couple_base * self.activite[idx] * (1.3 - 0.3 * stabilite) + self.rng.normal(0, 3))

        # Usure non plafonnée : au-delà du seuil personnel, le risque TWF continue de croître
        usure_precedente = self.usure_outil[idx - 1] + (p["malus_redemarrage_coupure"] if redemarre_apres_coupure else 0)
        facteur_poussiere = 1.0 + 0.004 * self.poussiere[idx] * p["sensibilite_poussiere"]
        facteur_tension = 1.0 + 0.6 * (1.0 - stabilite) * p["sensibilite_tension"]
        increment_usure = c.increment_usure_base * p["multiplicateur_usure"]
        increment_usure *= facteur_poussiere * facteur_tension
        self.usure_outil[idx] = usure_precedente + increment_usure
        encrassement = c.taux_base_encrassement * (self.poussiere[idx] / 100.0) * p["sensibilite_poussiere"]
        self.encrassement[idx] = min(self.encrassement[idx - 1] + encrassement, 100.0)
        roulements = c.taux_base_usure_roulements * ((self.vitesse_rotation[idx] / vitesse_base) ** 2) * p["multiplicateur_usure"]
        self.usure_roulements[idx] = min(self.usure_roulements[idx - 1] + roulements, 100.0)

        # Température : frottement, tension basse, usure et encrassement (moins de dissipation)
        temperature_cible = (
            self.ambiante[idx]
            + (c.facteur_temperature_frottement * (self.couple[idx] / 35.0))
            + p["decalage_temperature_cible"]
            + (ecart_tension * 0.15) * p["sensibilite_tension"]
            + c.facteur_chaleur_usure * (self.usure_outil[idx] / 100.0)
            + c.facteur_chaleur_encrassement * (self.encrassement[idx] / 100.0)
        )
        a = c.coef_lissage_temperature
        self.temperature_process[idx] = a * temperature_cible + (1 - a) * self.temperature_process[idx - 1]

        vibration = 0.4 * (self.vitesse_rotation[idx] / vitesse_base) + 0.4 * (1.0 - stabilite)
        if redemarre_apres_coupure:
            vibration += self.rng.uniform(3.0, 5.0) * p["multiplicateur_usure"]
        effet_usure = 4.5 * ((self.usure_outil[idx] / 100.0) ** 2)
        effet_roulements = c.coeff_vibration_roulements * ((self.usure_roulements[idx] / 100.0) ** 2)
        bruit = self.rng.normal(0, p["bruit_vibration"])
        self.vibration[idx] = np.clip(vibration + effet_usure + effet_roulements + bruit, 0.05, 15.0)

    def _calculer_risques_panne(self, idx: int) -> dict:
        """Risques concurrents du pas (AI4I 2020), chacun avec son précurseur ; PWF est tiré à part, au retour du courant."""
        c = self.config
        usure = self.usure_outil[idx]
        excedent = max(0.0, (usure - (self.seuil_use - c.bande_rampe_use)) / c.bande_rampe_use)
        twf = c.coeff_risque_use * (excedent**c.exposant_rampe_use) * self.params["multiplicateur_usure"]
        fenetre = min(idx, c.pas_fenetre_shd)  # HDF : surchauffe soutenue sur ~4 h
        surchauffe = self.temperature_process[idx - fenetre : idx + 1] - self.ambiante[idx - fenetre : idx + 1]
        hdf = c.coeff_risque_shd * max(0.0, float(surchauffe.mean()) - c.seuil_surchauffe_shd) ** 2
        terme_usure = (1.0 - c.poids_roulements_osf) * (usure / 100.0) ** 3 + c.poids_roulements_osf * (
            self.usure_roulements[idx] / 100.0
        ) ** 3
        osf = c.coeff_risque_sm * (self.couple[idx] / 45.0) * terme_usure
        return {"TWF": twf, "HDF": hdf, "OSF": osf, "RNF": c.risque_base_pa}

    def _immobiliser(self, debut: int, duree_heures: float, etat: str) -> int:
        """Arrêt de réparation ou d'entretien : refroidissement accéléré ; renvoie le premier pas de reprise."""
        fin = min(debut + 1 + int(np.ceil(duree_heures / self.pas_heures)), self.n)
        for idx in range(debut + 1, fin):
            self._arreter(idx, alpha_refroidissement=0.70, etat=etat)
        return fin

    def _gerer_phase_arret(self, index_panne: int) -> int:
        """Réparation : durée log-normale (médiane ≈ 3 h, jusqu'à 48 h) ; le composant en cause est remplacé."""
        duree = float(np.clip(self.rng.lognormal(mean=np.log(3.0), sigma=1.2), 0.5, 48.0))
        self.etat[index_panne] = "panne"
        fin = self._immobiliser(index_panne, duree, "panne")
        if fin < self.n:
            dernier, cause = fin - 1, self.mode_panne[index_panne]
            if cause == "TWF":
                self.usure_outil[dernier] = self.heures_outil[dernier] = 0.0
            if cause == "OSF":
                self.usure_roulements[dernier] = 0.0
            self.encrassement[dernier] = min(self.encrassement[dernier], self.config.residu_reinitialisation_encrassement)
            self.heures_maintenance[dernier] = 0.0
        return fin

    def _probabilite_maintenance_preventive(self, idx: int) -> float:
        score = max(self.usure_outil[idx], self.encrassement[idx], self.usure_roulements[idx]) / 100.0
        return self.config.probabilite_base_preventive + self.config.coeff_risque_preventif * score**2

    def _gerer_phase_maintenance_preventive(self, event_idx: int) -> int:
        """Visite préventive, plus courte qu'une réparation : outil changé, machine nettoyée, roulements révisés."""
        c = self.config
        duree = self.rng.lognormal(mean=np.log(c.heures_mediane_duree_preventive), sigma=0.6)
        duree = float(np.clip(duree, 0.5, c.heures_max_duree_preventive))
        fin = self._immobiliser(event_idx, duree, "entretien")
        if fin < self.n:
            dernier = fin - 1
            self.usure_outil[dernier] = self.heures_outil[dernier] = 0.0
            self.usure_roulements[dernier] *= 1.0 - c.reduction_roulements_preventive
            self.encrassement[dernier] = min(self.encrassement[dernier], c.residu_reinitialisation_encrassement)
            self.heures_maintenance[dernier] = 0.0
        return fin

    def evoluer_etats(self) -> pd.DataFrame:
        fin_immobilisation, alpha = 0, self.config.coef_lissage_temperature
        for idx in range(1, self.n):
            if idx < fin_immobilisation:
                continue
            if self.coupures[idx] == 1:  # délestage, avec un choc électrique possible au début de la coupure
                self._arreter(idx, alpha, etat="delestage")
                choc = self.config.probabilite_choc_coupure * self.params["sensibilite_panne_coupure"]
                if self.coupures[idx - 1] == 0 and self.rng.random() < choc:
                    self.failure[idx], self.mode_panne[idx] = 1, "PWF"
                    fin_immobilisation = self._gerer_phase_arret(idx)
                continue
            if self.activite[idx] == 0.0:  # arrêt planifié du week-end
                self._arreter(idx, alpha, etat="arret")
                continue

            self._mettre_a_jour_physique(idx, redemarre_apres_coupure=self.coupures[idx - 1] == 1)
            risques = self._calculer_risques_panne(idx)
            total = sum(risques.values())
            if self.rng.random() < min(total, 1.0):
                self.failure[idx] = 1
                self.mode_panne[idx] = self.rng.choice(list(risques.keys()), p=np.array(list(risques.values())) / total)
                fin_immobilisation = self._gerer_phase_arret(idx)
            elif self.rng.random() < self._probabilite_maintenance_preventive(idx):
                fin_immobilisation = self._gerer_phase_maintenance_preventive(idx)

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
