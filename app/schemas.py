from pydantic import BaseModel, Field

class MachineData(BaseModel):
    """Schéma d'entrée pour les données de la machine.
    """
    ambient_temperature: float = Field(..., description="Température ambiante en °C", example=25.5)
    process_temperature: float = Field(..., description="Température du process en °C", example=35.2)
    rotational_speed: float = Field(..., description="Vitesse de rotation (tr/min)", example=1500.0)
    torque: float = Field(..., description="Couple en Nm", example=40.5)
    tool_wear: float = Field(..., description="Usure de l'outil en minutes", example=10.0)
    machine_load: float = Field(..., description="Charge de la machine (0-1)", example=0.5)
    vibration: float = Field(..., description="Niveau de vibration", example=1.2)
    humidity: float = Field(..., description="Taux d'humidité en %", example=50.0)
    dust: float = Field(..., description="Concentration de poussière", example=0.1)
    voltage: float = Field(..., description="Tension électrique", example=220.0)
    voltage_stability: float = Field(..., description="Stabilité de la tension (0-1)", example=0.98)
    rain_flag: int = Field(..., ge=0, le=1, description="Indicateur de pluie (0/1)", example=0)
    power_outage: int = Field(..., ge=0, le=1, description="Coupure de courant (0/1)", example=0)
    season: int = Field(..., ge=0, le=1, description="Saison (Sèche, Pluvieuse)", example=0)
    machine_type: str = Field(..., description="Modèle de la machine", example="L")

class PredictionResponse(BaseModel):
    """Schéma de sortie renvoyé par l'API.
    """
    prediction: int = Field(..., description="0 pour Sain, 1 pour Panne")
    failure_probability: float = Field(..., description="Probabilité de panne entre 0 et 1")
    status: str = Field(default="success") 