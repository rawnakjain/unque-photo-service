"""Pydantic schemas for salon service extraction and fuzzy matching."""

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


class PriceTypeEnum(str, Enum):
    fixed = "FIXED"
    starting = "RANGE"


class ExtractedMenuItem(BaseModel):
    """Raw structured item extracted from salon price menu / catalogue."""
    service: str = Field(
        ...,
        description="The broad, base service category extracted from the menu (e.g., 'Waxing', 'Facial', 'Haircut')",
    )
    variation_name: str = Field(
        ...,
        description="The specific purchasable item including meaningful modifiers like packages, brands, body areas, hair lengths, or techniques (e.g., 'Full Leg Wax', 'Root Touchup - Inoa')",
    )
    canonical_service_name: Optional[str] = Field(
        None,
        description="Map the extracted service to the most appropriate match from the allowed whitelist, or null if it does not fit",
    )
    recommended_new_service_name: Optional[str] = Field(
        None,
        description="If a service does NOT fit neatly into any whitelisted category, provide a clean, standardized candidate string here. Otherwise, null",
    )
    price: Optional[float] = Field(
        None,
        description="Clean numeric value (integer/float) without currency symbols, or null if unreadable. For price ranges, use the lower bound.",
    )
    price_raw: Optional[str] = Field(
        None,
        description="Exact price string as it appears in the image (e.g., '₹500', 'Starting @ 500')",
    )
    price_type: PriceTypeEnum = Field(
        default=PriceTypeEnum.fixed,
        description="Must be 'fixed' or 'starting'. Use 'starting' if prefixed with terms like 'Starts from', 'Onwards', 'From', or if it represents a price range/base tier. Otherwise, use 'fixed'",
    )
    approx_duration: Optional[str] = Field(
        None,
        description="Printed duration (e.g., '45 mins') or null",
    )
    gender: GenderEnum = Field(
        default=GenderEnum.U,
        description="Exactly one of 'M', 'F', or 'U' (unisex/general)",
    )
    person_type: PersonTypeEnum = Field(
        default=PersonTypeEnum.Adult,
        description="Exactly one of 'Adult' or 'Child'",
    )


class GeminiExtractionResponse(BaseModel):
    """Container for all items extracted by Gemini from the image."""
    items: list[ExtractedMenuItem] = Field(
        default_factory=list,
        description="List of menu items extracted from the image",
    )


class ServiceRecommendation(BaseModel):
    """Recommended service candidate retrieved from MongoDB."""
    serviceId: str
    serviceName: str
    score: Optional[float] = None
    similarity_percentage: Optional[int] = None


class VariationRecommendation(BaseModel):
    """Recommended variation candidate retrieved from MongoDB."""
    variationId: Optional[str] = None
    variationName: str
    score: Optional[float] = None
    similarity_percentage: Optional[int] = None


class ProcessedMenuItem(BaseModel):
    """An extracted menu item together with top-5 database recommendations for both service and variation."""
    extracted: ExtractedMenuItem
    service_recommendations: list[ServiceRecommendation] = Field(
        default_factory=list,
        description="Top 5 recommended service matches from salonservices collection",
    )
    variation_recommendations: list[VariationRecommendation] = Field(
        default_factory=list,
        description="Top 5 recommended variation matches from salonservicevariations collection",
    )
    # For backwards compatibility - combined top recommendations
    recommendations: list[VariationRecommendation] = Field(
        default_factory=list,
        description="Combined top recommendations (deprecated, use service_recommendations/variation_recommendations)",
    )
    selected_service_id: Optional[str] = Field(
        None,
        description="The serviceId chosen by the salon owner (defaults to top service recommendation)",
    )
    selected_service_name: Optional[str] = Field(
        None,
        description="The service name chosen by the salon owner",
    )
    selected_variation_id: Optional[str] = Field(
        None,
        description="The variationId chosen by the salon owner (defaults to top variation recommendation)",
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
    stage_timings: Optional[dict] = None


class VariationSearchRequest(BaseModel):
    """Request to search/query MongoDB variations dynamically."""
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    service_id: Optional[str] = Field(default=None, description="Optional serviceId to restrict variation search to a specific service")


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


class VariationRecommendationItem(BaseModel):
    """Single variation recommendation with index."""
    index: int
    name: str
    score: Optional[float] = None


class RecommendedVariation(BaseModel):
    """Recommended variation with service info and nested variation recommendations."""
    variationName: str
    serviceName: str
    serviceId: str
    price: Optional[float] = None
    priceType: PriceTypeEnum = PriceTypeEnum.fixed
    variationRecommendations: list[VariationRecommendationItem] = Field(default_factory=list)


class MediaServiceItem(BaseModel):
    """Container for recommended variations per extracted item."""
    recommendedVariations: list[RecommendedVariation] = Field(default_factory=list)


class MediaServiceResponse(BaseModel):
    """Response for media service image-to-text endpoint."""
    data: list[MediaServiceItem] = Field(default_factory=list)
    errors: dict = Field(default_factory=dict)
    stage_timings: Optional[dict] = None
