from typing import Optional

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
