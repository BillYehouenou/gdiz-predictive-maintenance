from datetime import date, datetime
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field


class DonneesMachine(BaseModel):
    """Schéma d'entrée pour les données de la machine."""

    id_machine: Optional[str] = Field(default=None, description="ID de la machine", examples=["GDIZ_TEX_M_01"])
    temperature_ambiante: float = Field(..., description="Température ambiante en °C", examples=[28.0])
    temperature_process: float = Field(..., description="Température du process en °C", examples=[52.0])
    vitesse_rotation: float = Field(..., description="Vitesse de rotation (tr/min)", examples=[1200.0])
    couple: float = Field(..., description="Couple en Nm", examples=[35.0])
    niveau_activite: float = Field(..., ge=0.0, le=1.0, description="Taux d'activité de la machine (0-1)", examples=[0.6])
    vibration: float = Field(..., description="Niveau de vibration (mm/s)", examples=[1.5])
    humidite: float = Field(..., description="Humidité relative (%)", examples=[60.0])
    concentration_poussiere: float = Field(..., description="Concentration de poussière (µg/m³)", examples=[120.0])
    niveau_tension: float = Field(..., description="Tension réseau SBEE (V)", examples=[222.0])
    precipitation_mm: float = Field(..., ge=0.0, description="Précipitations sur le dernier pas de 30 min (mm)", examples=[0.0])
    indicateur_coupure: int = Field(..., ge=0, le=1, description="Délestage SBEE (0/1)", examples=[0])
    type_machine: str = Field(..., description="Qualité de machine (L, M, H)", examples=["M"])
    heures_depuis_changement_outil: float = Field(
        ..., ge=0.0, description="Heures de fonctionnement depuis le dernier changement d'outil (GMAO)", examples=[2400.0]
    )
    heures_depuis_maintenance: float = Field(
        ..., ge=0.0, description="Heures depuis la dernière intervention de maintenance (GMAO)", examples=[600.0]
    )


class ReponsePrediction(BaseModel):
    """Schéma de sortie renvoyé par l'API."""

    prediction: int = Field(..., description="0 pour Sain, 1 pour Panne")
    probabilite_panne: float = Field(..., description="Probabilité de panne entre 0 et 1")
    status: str = Field(default="success")


class InfoModele(BaseModel):
    """Schéma de sortie de /api/v1/info — version et seuil du modèle actuellement chargé."""

    model_name: str = Field(..., description="Nom du modèle enregistré dans MLflow")
    model_version: str = Field(..., description="Version @champion actuellement chargée en mémoire")
    threshold: float = Field(..., description="Seuil de décision courant")


class ReponseRechargement(BaseModel):
    """Schéma de sortie de /api/v1/reload."""

    model_version: str = Field(..., description="Version @champion chargée après rechargement")
    threshold: float = Field(..., description="Seuil de décision après rechargement")
    status: str = Field(default="success")


class Indicateurs(BaseModel):
    disponibilite: Optional[float] = Field(description="Part du temps ouvert sans panne (hors arrêts planifiés et délestages)")
    mtbf_h: Optional[float] = Field(description="Heures de marche entre deux pannes")
    cout: float = Field(description="Coût des pannes en FCFA, durée réelle des réparations comprise")
    heures_arret: float
    heures_marche: float
    pannes: int


class Periode(BaseModel):
    code: str
    libelle: str
    libelle_precedente: Optional[str]


class PannesGammeCause(BaseModel):
    gamme: str
    cause: str = Field(description="TWF usure, HDF surchauffe, OSF surcharge, PWF électrique, RNF aléatoire")
    pannes: int


class PointCout(BaseModel):
    debut: date
    cout: float


class TableauParc(BaseModel):
    periode: Periode
    indicateurs: Indicateurs
    indicateurs_precedents: Optional[Indicateurs]
    pannes_par_gamme: list[PannesGammeCause]
    cout_serie: list[PointCout]
    cout_serie_precedente: list[PointCout]


Statut = Literal["normal", "surveillance", "panne_predite", "en_panne"]


class EtatMachine(BaseModel):
    id_machine: str
    numero: str
    nom: str = Field(description="Cardeuse, continu à filer, métier à tisser ou rameuse")
    gamme: str
    statut: Statut
    risque: float = Field(description="Probabilité de panne à 5 jours")
    criticite: float = Field(description="Part de production perdue si la machine s'arrête, sur 10")
    zone: Optional[Literal["surveiller", "planifier", "agir"]]
    cause: Optional[str] = Field(description="Famille physique dominante, si en alerte")
    sante: int


class EtatParc(BaseModel):
    fin_donnees: datetime
    seuil: float
    machines: list[EtatMachine]
    comptes: dict[str, int]


class Contribution(BaseModel):
    feature: str
    libelle: str
    unite: str
    famille: Optional[str]
    valeur: Optional[Union[float, str]]
    contribution: float = Field(description="Contribution SHAP vers la panne (log-odds)")


class EvenementPanne(BaseModel):
    date: datetime
    cause: str
    duree_h: float
    cout: float


class Capteur(BaseModel):
    capteur: str
    libelle: str
    unite: str
    valeur: Optional[float]
    limite: float = Field(description="99e centile du parc (1er pour la tension) sur la période d'entraînement")
    hors_limite: bool


class FicheMachine(EtatMachine):
    seuil: float
    contributions: list[Contribution]
    historique: list[EvenementPanne]
    douze_mois: Indicateurs
    capteurs_30j: list[Capteur]
    pannes_30j: int


class PointRisque(BaseModel):
    timestamp: datetime
    risque: float
    delestage: bool
    panne: bool
