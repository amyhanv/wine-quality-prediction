from pydantic import BaseModel, Field, ValidationError

class WineData(BaseModel):
    fixed_acidity: float = Field(..., ge=0)
    volatile_acidity: float = Field(..., ge=0)
    citric_acid: float = Field(..., ge=0)
    alcohol: float = Field(..., ge=8.0, le=15.0)
    quality: int = Field(..., ge=3, le=8)

# Example input
try:
    wine = WineData(
        fixed_acidity=7.4,
        volatile_acidity=0.7,
        citric_acid=0.0,
        alcohol=9.4,
        quality=5
    )
    print("Validated data:", wine.model_dump())
except ValidationError as e:
    print("Validation error:", e)
