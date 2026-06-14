from pydantic import BaseModel, Field


class MachineData(BaseModel):
    """Schéma d'entrée pour les données de la machine."""

    ambient_temperature: float = Field(..., description="Température ambiante en °C", examples=[28.0])
    process_temperature: float = Field(..., description="Température du process en °C", examples=[52.0])
    rotational_speed: float = Field(..., description="Vitesse de rotation (tr/min)", examples=[1200.0])
    torque: float = Field(..., description="Couple en Nm", examples=[35.0])
    tool_wear: float = Field(..., description="Usure de l'outil (%)", examples=[30.0])
    activity_level: float = Field(..., ge=0.0, le=1.0, description="Taux d'activité de la machine (0-1)", examples=[0.6])
    vibration: float = Field(..., description="Niveau de vibration (mm/s)", examples=[1.5])
    humidity: float = Field(..., description="Humidité relative (%)", examples=[60.0])
    dust_concentration: float = Field(..., description="Concentration de poussière (µg/m³)", examples=[120.0])
    voltage_level: float = Field(..., description="Tension réseau SBEE (V)", examples=[222.0])
    voltage_stability: float = Field(..., description="Stabilité de la tension (%)", examples=[75.0])
    rain_flag: int = Field(..., ge=0, le=1, description="Indicateur de pluie (0/1)", examples=[0])
    power_loss_indicator: int = Field(..., ge=0, le=1, description="Délestage SBEE (0/1)", examples=[0])
    benin_season: int = Field(..., ge=0, le=4, description="Saison béninoise (0=Saison pluvieuse, 1=Saison sèche)", examples=[1])
    machine_type: str = Field(..., description="Qualité de machine (L, M, H)", examples=["M"])


class PredictionResponse(BaseModel):
    """Schéma de sortie renvoyé par l'API."""

    prediction: int = Field(..., description="0 pour Sain, 1 pour Panne")
    failure_probability: float = Field(..., description="Probabilité de panne entre 0 et 1")
    status: str = Field(default="success")
