from pydantic import BaseModel, Field

class MachineData(BaseModel):
    air_temperature: float = Field(..., example=298.1, description="Température ambiante en K")
    process_temperature: float = Field(..., example=308.6, description="Température du process en K")
    rotational_speed: float = Field(..., example=1500, description="Vitesse de rotation en rpm")
    torque: float = Field(..., example=40.0, description="Couple en Nm")
    tool_wear: float = Field(..., example=0, description="Usure de l'outil en minutes")

    class Config:
        schema_extra = {
            "example": {
                "air_temperature": 298.1,
                "process_temperature": 308.6,
                "rotational_speed": 1551,
                "torque": 42.8,
                "tool_wear": 0
            }
        }

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    status: str