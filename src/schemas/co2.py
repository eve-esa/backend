from typing import Optional
from pydantic import BaseModel, Field


class CO2EquivalenceComparison(BaseModel):
    """Schema for CO2 equivalence comparison items."""
    
    title: str = Field(..., description="Title of the comparison item")
    co2eq_kg: float = Field(..., description="CO2 equivalent in kilograms", ge=0)
    unit_singular: str = Field(..., description="Singular form of the unit")
    unit_plural: str = Field(..., description="Plural form of the unit")
    enabled: bool = Field(default=True, description="Whether this comparison is enabled")


class CO2EquivalenceResult(BaseModel):
    """Schema for CO2 equivalence calculation result."""
    
    title: str = Field(..., description="Title of the comparison item")
    co2eq_kg: float = Field(..., description="Base CO2 equivalent in kilograms")
    equivalent_count: Optional[int] = Field(default=None, description="Number of equivalent items")
    text: str = Field(..., description="Formatted equivalence text")

