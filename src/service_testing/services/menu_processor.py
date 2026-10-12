"""Orchestrator combining Gemini Vision extraction and MongoDB Atlas Fuzzy Search for services and variations."""

import logging
import time
from typing import Optional

from service_testing.schemas import (
    MenuProcessingResponse,
    ProcessedMenuItem,
    ServiceRecommendation,
    VariationRecommendation,
)
from service_testing.services.fuzzy_search import FuzzySearchService
from service_testing.services.gemini_extractor import GeminiExtractorService

logger = logging.getLogger(__name__)

MIN_CONFIDENCE = 80


class MenuProcessorService:
    def __init__(
        self,
        gemini_extractor: Optional[GeminiExtractorService] = None,
        fuzzy_search: Optional[FuzzySearchService] = None,
    ):
        self.gemini_extractor = gemini_extractor or GeminiExtractorService()
        self.fuzzy_search = fuzzy_search or FuzzySearchService()

    def process_menu_image(
        self,
        image_bytes: bytes,
        filename: str,
        mime_type: str = "image/jpeg",
        top_k_recommendations: int = 5,
    ) -> MenuProcessingResponse:
        """
        1. Calls Gemini to extract items from image (includes canonical_service_name from whitelist).
        2. For each extracted item, determine the service lookup name using priority:
           canonical_service_name > recommended_new_service_name > service
        3. Fuzzy search service in salonservices collection to get serviceId.
        4. Fuzzy search variation in salonservicevariations collection (global, not restricted by service).
        5. Applies 80% confidence threshold to both services and variations.
        6. Returns items with serviceId, service name, and variation recommendations (separate lists).
        """
        start_time = time.time()

        extraction_res = self.gemini_extractor.extract_from_image(
            image_bytes=image_bytes,
            mime_type=mime_type,
        )

        processed_items: list[ProcessedMenuItem] = []

        for item in extraction_res.items:
            # Determine service lookup name using priority:
            # 1. canonical_service_name (whitelist match)
            # 2. recommended_new_service_name (new service candidate)
            # 3. service (extracted base category)
            service_lookup_name = (
                item.canonical_service_name
                or item.recommended_new_service_name
                or item.service
            )

            # Step 1: Fuzzy Search for service in salonservices collection to get serviceId
            service_recs: list[ServiceRecommendation] = []
            if service_lookup_name:
                try:
                    service_recs = self.fuzzy_search.recommend_services(
                        query=service_lookup_name,
                        top_k=top_k_recommendations,
                        min_confidence=MIN_CONFIDENCE,
                    )
                except Exception as e:
                    logger.error(f"Error searching services for '{service_lookup_name}': {e}")

            # Select top service match - serviceId comes ONLY from service recommendations
            top_service = service_recs[0] if service_recs else None
            selected_service_id = top_service.serviceId if top_service else None
            selected_service_name = top_service.serviceName if top_service else service_lookup_name

            # Step 2: Fuzzy Search for variation in salonservicevariations collection (global search)
            variation_query = item.variation_name.strip()
            variation_recs: list[VariationRecommendation] = []
            if variation_query:
                try:
                    variation_recs = self.fuzzy_search.recommend_variations(
                        query=variation_query,
                        top_k=top_k_recommendations,
                        min_confidence=MIN_CONFIDENCE,
                    )
                except Exception as e:
                    logger.error(f"Error searching variations for '{variation_query}': {e}")

            # Select top variation match
            top_variation = variation_recs[0] if variation_recs else None
            selected_variation_id = top_variation.variationId if top_variation else None
            selected_variation_name = top_variation.variationName if top_variation else variation_query

            # No combined recommendations - keep service and variation separate
            # Provide fallback only if no recommendations at all
            if not service_recs and not variation_recs:
                logger.warning(
                    f"No recommendations above {MIN_CONFIDENCE}% confidence for "
                    f"service='{service_lookup_name}', variation='{variation_query}'"
                )

            processed_items.append(
                ProcessedMenuItem(
                    extracted=item,
                    service_recommendations=service_recs,
                    variation_recommendations=variation_recs,
                    recommendations=[],  # Deprecated - keep empty
                    selected_service_id=selected_service_id,
                    selected_service_name=selected_service_name,
                    selected_variation_id=selected_variation_id,
                    selected_variation_name=selected_variation_name,
                )
            )

        duration = time.time() - start_time

        return MenuProcessingResponse(
            image_filename=filename,
            total_items=len(processed_items),
            items=processed_items,
            processing_time_seconds=round(duration, 2),
            gemini_model_used=self.gemini_extractor.model_name,
        )
