"""Pydantic schemas for salon service extraction and hybrid matching."""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class GenderEnum(str, Enum):
    M = "M"
    F = "F"
    U = "U"


class PersonTypeEnum(str, Enum):
    Adult = "Adult"
    Child = "Child"
    Any = "Any"


class ExtractedMenuItem(BaseModel):
    """Raw structured item extracted from salon price menu / catalogue."""
    service: str = Field(
        ...,
        description="Standard service category matched to predefined list (e.g., haircut, facial, waxing, etc.)",
    )
    service_variation: str = Field(
        ...,
        description="Specific service variation as printed on the menu (e.g., 'Signature Haircut - BOB by Senior Stylist', 'Full Arm Wax')",
    )
    price: Optional[float] = Field(
        None,
        description="Price in numeric format if available",
    )
    price_raw: Optional[str] = Field(
        None,
        description="Original price string from menu if present (e.g., '₹850', '$45')",
    )
    approx_duration: Optional[str] = Field(
        None,
        description="Approximate duration (e.g., '45 mins', '1 hr')",
    )
    gender: GenderEnum = Field(
        default=GenderEnum.U,
        description="Target gender: M (Male), F (Female), U (Unisex)",
    )
    person_type: PersonTypeEnum = Field(
        default=PersonTypeEnum.Adult,
        description="Target audience: Adult, Child, or Any",
    )


class GeminiExtractionResponse(BaseModel):
    """Container for all items extracted by Gemini from the image."""
    items: list[ExtractedMenuItem] = Field(
        default_factory=list,
        description="List of menu items extracted from the image",
    )


class VariationRecommendation(BaseModel):
    """Recommended variation candidate retrieved from MongoDB."""
    serviceId: Optional[str] = None
    variationId: Optional[str] = None
    variationName: str
    score: Optional[float] = None
    similarity_percentage: Optional[int] = None


class ProcessedMenuItem(BaseModel):
    """An extracted menu item together with top-5 database recommendations."""
    extracted: ExtractedMenuItem
    recommendations: list[VariationRecommendation] = Field(
        default_factory=list,
        description="Top 5 recommended matches from MongoDB hybrid RAG",
    )
    selected_variation_id: Optional[str] = Field(
        None,
        description="The variationId chosen by the salon owner (defaults to top recommendation)",
    )
    selected_variation_name: Optional[str] = Field(
        None,
        description="The variationName chosen by the salon owner",
    )


class MenuProcessingResponse(BaseModel):
    """Overall response returned by extract-and-match endpoint."""
    image_filename: str
    total_items: int
    items: list[ProcessedMenuItem]
    processing_time_seconds: float
    gemini_model_used: str


class VariationSearchRequest(BaseModel):
    """Request to search/query MongoDB variations dynamically."""
    query: str
    top_k: int = Field(default=5, ge=1, le=20)


class ConfirmedItemPayload(BaseModel):
    service: str
    extracted_variation: str
    price: Optional[float] = None
    price_raw: Optional[str] = None
    duration: Optional[str] = None
    gender: GenderEnum
    person_type: PersonTypeEnum
    matched_service_id: Optional[str] = None
    matched_variation_id: Optional[str] = None
    matched_variation_name: str


class CatalogConfirmationRequest(BaseModel):
    """Payload when salon owner confirms their onboarding selections."""
    salon_name: Optional[str] = "My Salon"
    items: list[ConfirmedItemPayload]


class CatalogConfirmationResponse(BaseModel):
    status: str
    total_confirmed: int
    message: str
    confirmed_items: list[ConfirmedItemPayload]
