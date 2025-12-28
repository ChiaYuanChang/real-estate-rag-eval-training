from enum import StrEnum
from typing import List

from pydantic import BaseModel, Field


class PropertyTypeEnum(StrEnum):
    townhouse = "townhouse"
    condo = "condo"


# 2. 定義 Cypher 變數模型 (硬性指標)
class CypherVariables(BaseModel):
    city: str | None = Field(
        default=None,
        description="The specific city name extracted from the user query (e.g., '高雄市'). Set to null if not mentioned."
    )
    district: str | None = Field(
        default=None,
        description="The specific district name (e.g., '楠梓區'). Set to null if not mentioned."
    )
    street: str | None = Field(
        default=None,
        description="The specific street name (e.g., '右昌街'). Set to null if not mentioned."
    )
    min_price: int | None = Field(
        default=None,
        description="Minimum budget in TWD (New Taiwan Dollar). Must be an integer (e.g., 10000000 for 1000萬)."
    )
    max_price: int | None = Field(
        default=None,
        description="Maximum budget in TWD. Must be an integer."
    )
    min_interior_area: float | None = Field(
        default=None,
        description="Minimum interior area size in 'Ping' (坪). Use float for precision."
    )
    min_bedroom: int | None = Field(
        default=None,
        description="Minimum number of bedrooms required."
    )
    min_bathroom: int | None = Field(
        default=None,
        description="Minimum number of bathrooms required."
    )
    property_type: PropertyTypeEnum | None = Field(
        default=None,
        description="The type of property. strict mapping: '透天/別墅' -> 'townhouse', '大樓/公寓' -> 'condo'."
    )
    min_age: int | None = Field(
        default=None,
        description="Minimum property age in years."
    )
    max_age: int | None = Field(
        default=None,
        description="Maximum property age in years."
    )


class RealEstateQuery(BaseModel):
    cypher_variables: CypherVariables = Field(
        description="Structured variables for database filtering (Hard Filters)."
    )
    abstract_requirements: List[str] = Field(
        default_factory=list,
        description="A list of abstract requirements, stylistic preferences, or facility needs (Soft Filters) in Traditional Chinese (e.g., ['開放式廚房', '採光好'])."
    )
