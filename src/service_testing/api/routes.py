"""FastAPI API routes for salon menu extraction and fuzzy search (variations only)."""

import logging
from typing import Optional
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from service_testing.schemas import (
    MenuProcessingResponse,
    VariationRecommendation,
    MediaServiceResponse,
    MediaServiceItem,
    RecommendedVariation,
    VariationRecommendationItem,
)
from service_testing.services.menu_processor import MenuProcessorService
from service_testing.services.image_quality import ImageQualityError, check_image_quality_or_raise

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Salon Menu POC"])

# Lazy-loaded singleton services
_menu_processor: Optional[MenuProcessorService] = None


def get_menu_processor() -> MenuProcessorService:
    global _menu_processor
    if _menu_processor is None:
        _menu_processor = MenuProcessorService()
    return _menu_processor


@router.post("/menu/extract-and-match", response_model=MenuProcessingResponse)
async def extract_and_match_menu(
    file: UploadFile = File(...),
):
    """
    Upload a menu image/catalogue.
    1. Extracts services & variations with Gemini Flash (services mapped to whitelist).
    2. Performs Atlas Fuzzy Search on salonservicevariations for variations only.
    3. Applies 80% confidence threshold.
    4. Returns structured items with canonical_service_name and variation recommendations.
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

        # Validate image quality (blur/corruption detection)
        try:
            check_image_quality_or_raise(content)
        except ImageQualityError as iqe:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(iqe),
            )

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


@router.post("/v1/media-service/image-2-text/photo", response_model=MediaServiceResponse)
async def media_service_image_to_text(
    file: UploadFile = File(...),
):
    """
    Media Service endpoint: Upload a menu image/catalogue.
    1. Extracts services & variations with Gemini Flash.
    2. Matches services to salonservices collection (gets serviceId, serviceName).
    3. Matches variations to salonservicevariations collection.
    4. Returns nested structure with recommendedVariations and variationRecommendations.
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

        # Validate image quality (blur/corruption detection)
        try:
            check_image_quality_or_raise(content)
        except ImageQualityError as iqe:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(iqe),
            )

        mime_type = file.content_type or "image/jpeg"
        processor = get_menu_processor()

        # Process using existing logic
        result = processor.process_menu_image(
            image_bytes=content,
            filename=file.filename or "uploaded_menu.jpg",
            mime_type=mime_type,
            top_k_recommendations=5,
        )

        # Transform to new response format
        media_items: list[MediaServiceItem] = []

        for item in result.items:
            # Get top service match
            top_service = item.service_recommendations[0] if item.service_recommendations else None
            service_id = top_service.serviceId if top_service else ""
            service_name = top_service.serviceName if top_service else item.selected_service_name or ""

            # Get top variation match
            top_variation = item.variation_recommendations[0] if item.variation_recommendations else None
            variation_name = top_variation.variationName if top_variation else item.selected_variation_name or item.extracted.variation_name

            # Build variation recommendations with index
            var_recs = [
                VariationRecommendationItem(
                    index=i,
                    name=vr.variationName,
                    score=vr.score,
                )
                for i, vr in enumerate(item.variation_recommendations)
            ]

            recommended_variation = RecommendedVariation(
                variationName=variation_name,
                serviceName=service_name,
                serviceId=service_id,
                price=item.extracted.price,
                priceType=item.extracted.price_type,
                variationRecommendations=var_recs,
            )

            media_items.append(MediaServiceItem(recommendedVariations=[recommended_variation]))

        return MediaServiceResponse(data=media_items, errors={})

    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.exception("Media service image processing error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process menu image: {str(e)}",
        )
