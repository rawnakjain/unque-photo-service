"""FastAPI API routes for salon menu extraction and hybrid search."""

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
from service_testing.services.menu_processor import MenuProcessorService
from service_testing.services.mongo_search import MongoSearchService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Salon Menu POC"])

# Lazy-loaded singleton services
_mongo_service: Optional[MongoSearchService] = None
_menu_processor: Optional[MenuProcessorService] = None


def get_mongo_service() -> MongoSearchService:
    global _mongo_service
    if _mongo_service is None:
        _mongo_service = MongoSearchService()
    return _mongo_service


def get_menu_processor() -> MenuProcessorService:
    global _menu_processor
    if _menu_processor is None:
        _menu_processor = MenuProcessorService(mongo_search=get_mongo_service())
    return _menu_processor


@router.get("/health")
def health_check():
    """Returns system status including MongoDB connection and Gemini configuration."""
    mongo = get_mongo_service()
    mongo_status = mongo.check_connection()

    return {
        "status": "healthy" if mongo_status.get("status") == "connected" else "degraded",
        "gemini": {
            "configured": bool(settings.gemini_api_key),
            "model": settings.gemini_model,
        },
        "mongodb": mongo_status,
        "standard_services_count": len(UNIQUE_STANDARD_SERVICES),
    }


@router.get("/services")
def get_standard_services() -> list[str]:
    """Returns the predefined ~100 standard salon service categories."""
    return UNIQUE_STANDARD_SERVICES


@router.post("/variations/search", response_model=list[VariationRecommendation])
def search_variations(payload: VariationSearchRequest):
    """
    Search variations directly in MongoDB Atlas using hybrid search ($rankFusion).
    Returns top-k recommendations.
    """
    try:
        mongo = get_mongo_service()
        results = mongo.recommend_variations(query=payload.query, top_k=payload.top_k)
        return results
    except Exception as e:
        logger.error(f"Error executing variation search: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hybrid search failed: {str(e)}",
        )


@router.post("/menu/extract-and-match", response_model=MenuProcessingResponse)
async def extract_and_match_menu(
    file: UploadFile = File(...),
):
    """
    Upload a menu image/catalogue.
    1. Extracts services & variations with Gemini Flash.
    2. Performs Atlas Hybrid Search for each variation.
    3. Returns structured items with top-5 recommended database matches.
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
