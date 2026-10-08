"""FastAPI API routes for salon menu extraction and fuzzy search."""

import logging
from typing import Optional
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from service_testing.config import settings
from service_testing.constants import UNIQUE_STANDARD_SERVICES
from service_testing.schemas import (
    CatalogConfirmationRequest,
    CatalogConfirmationResponse,
    MenuProcessingResponse,
    VariationRecommendation,
    VariationSearchRequest,
)
from service_testing.services.fuzzy_search import FuzzySearchService
from service_testing.services.menu_processor import MenuProcessorService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Salon Menu POC"])

# Lazy-loaded singleton services
_fuzzy_service: Optional[FuzzySearchService] = None
_menu_processor: Optional[MenuProcessorService] = None


def get_fuzzy_service() -> FuzzySearchService:
    global _fuzzy_service
    if _fuzzy_service is None:
        _fuzzy_service = FuzzySearchService()
    return _fuzzy_service


def get_menu_processor() -> MenuProcessorService:
    global _menu_processor
    if _menu_processor is None:
        _menu_processor = MenuProcessorService(fuzzy_search=get_fuzzy_service())
    return _menu_processor


@router.get("/health")
def health_check():
    """Returns system status including MongoDB connection and Gemini configuration."""
    fuzzy = get_fuzzy_service()
    fuzzy_status = fuzzy.check_connection()

    return {
        "status": "healthy" if fuzzy_status.get("status") == "connected" else "degraded",
        "gemini": {
            "configured": bool(settings.gemini_api_key),
            "model": settings.gemini_model,
        },
        "mongodb": fuzzy_status,
        "standard_services_count": len(UNIQUE_STANDARD_SERVICES),
    }


@router.get("/services")
def get_standard_services() -> list[str]:
    """Returns the predefined ~100 standard salon service categories."""
    return UNIQUE_STANDARD_SERVICES


@router.post("/variations/search", response_model=list[VariationRecommendation])
def search_variations(payload: VariationSearchRequest):
    """
    Search variations directly in MongoDB Atlas using fuzzy search.
    Returns top-k recommendations with confidence >= 80%.
    """
    try:
        fuzzy = get_fuzzy_service()
        results = fuzzy.recommend_variations(query=payload.query, top_k=payload.top_k, min_confidence=80)
        return results
    except Exception as e:
        logger.error(f"Error executing variation search: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Fuzzy search failed: {str(e)}",
        )


@router.post("/services/search", response_model=list[VariationRecommendation])
def search_services(payload: VariationSearchRequest):
    """
    Search services directly in MongoDB Atlas using fuzzy search on salonservices collection.
    Returns top-k recommendations with confidence >= 80%.
    """
    try:
        fuzzy = get_fuzzy_service()
        results = fuzzy.recommend_services(query=payload.query, top_k=payload.top_k, min_confidence=80)
        return results
    except Exception as e:
        logger.error(f"Error executing service search: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Fuzzy search failed: {str(e)}",
        )


@router.post("/menu/extract-and-match", response_model=MenuProcessingResponse)
async def extract_and_match_menu(
    file: UploadFile = File(...),
):
    """
    Upload a menu image/catalogue.
    1. Extracts services & variations with Gemini Flash.
    2. Performs Atlas Fuzzy Search on salonservices (for service) and salonservicevariations (for variation).
    3. Applies 80% confidence threshold.
    4. Returns structured items with top-5 recommended database matches.
    """
    if not file.content_type or not (
        file.content_type.startswith("image/") or file.filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be an image (PNG, JPEG, WebP).",
        )

    try:
        content = await file.read()
        mime_type = file.content_type or "image/jpeg"
        processor = get_menu_processor()

        response = processor.process_menu_image(
            image_bytes=content,
            filename=file.filename or "uploaded_menu.jpg",
            mime_type=mime_type,
            top_k_recommendations=5,
        )
        return response

    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.exception("Menu processing error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process menu image: {str(e)}",
        )


@router.post("/menu/confirm", response_model=CatalogConfirmationResponse)
def confirm_catalog(payload: CatalogConfirmationRequest):
    """
    Salon owner confirms the selected variation mappings.
    Returns summary confirmation of onboarded catalog items.
    """
    logger.info(f"Salon '{payload.salon_name}' confirmed {len(payload.items)} items.")
    return CatalogConfirmationResponse(
        status="success",
        total_confirmed=len(payload.items),
        message=f"Successfully onboarded {len(payload.items)} services for {payload.salon_name}!",
        confirmed_items=payload.items,
    )
