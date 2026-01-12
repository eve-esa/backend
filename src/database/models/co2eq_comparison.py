from typing import ClassVar, Optional
from pydantic import Field
from src.database.mongo_model import MongoModel


class CO2EQComparison(MongoModel):
    """Model for storing CO2 equivalence comparison items."""

    title: str = Field(..., description="Title of the comparison item")
    co2eq_kg: float = Field(..., description="CO2 equivalent in kilograms", ge=0)
    unit_singular: str = Field(..., description="Singular form of the unit")
    unit_plural: str = Field(..., description="Plural form of the unit")
    enabled: bool = Field(default=True, description="Whether this comparison is enabled")

    collection_name: ClassVar[str] = "co2eq_comparisons"

