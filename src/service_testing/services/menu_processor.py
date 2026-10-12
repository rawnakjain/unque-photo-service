"""Orchestrator combining Gemini Vision extraction and MongoDB Atlas Fuzzy Search for services and variations."""

import logging
import time
from typing import Optional

from service_testing.schemas import (
    MenuProcessingResponse,
    ProcessedMenuItem,
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
        2. For each extracted item, fuzzy search service in salonservices collection to get serviceId.
        3. For each extracted item, fuzzy search variation in salonservicevariations collection.
        4. Applies 80% confidence threshold to both services and variations.
        5. Returns items with serviceId, service name, and variation recommendations.
        """
        start_time = time.time()

        extraction_res = self.gemini_extractor.extract_from_image(
            image_bytes=image_bytes,
            mime_type=mime_type,
        )

        processed_items: list[ProcessedMenuItem] = []

        for item in extraction_res.items:
            # Service name from Gemini (canonical_service_name from whitelist, fallback to service)
            canonical_name = item.canonical_service_name or item.service

            # Step 1: Fuzzy Search for service in salonservices collection to get serviceId
            service_recs: list[VariationRecommendation] = []
            if canonical_name:
                try:
                    service_recs = self.fuzzy_search.recommend_services(
                        query=canonical_name,
                        top_k=top_k_recommendations,
                        min_confidence=MIN_CONFIDENCE,
                    )
                except Exception as e:
                    logger.error(f"Error searching services for '{canonical_name}': {e}")

            # Step 2: Fuzzy Search for variation in salonservicevariations collection
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

            # Combined recommendations for backwards compatibility
            all_recommendations = service_recs + variation_recs
            all_recommendations.sort(key=lambda x: x.similarity_percentage or 0, reverse=True)
            combined_recs = all_recommendations[:top_k_recommendations]

            # If no recommendations meet confidence threshold, provide fallback
            if not combined_recs:
                logger.warning(f"No recommendations above {MIN_CONFIDENCE}% confidence for service='{canonical_name}', variation='{variation_query}'")
                combined_recs = [
                    VariationRecommendation(
                        variationName=variation_query or canonical_name,
                        score=0.0,
                        similarity_percentage=0,
                    )
                ]

            top_service = service_recs[0] if service_recs else None
            top_variation = variation_recs[0] if variation_recs else None
            top_combined = combined_recs[0] if combined_recs else None

            processed_items.append(
                ProcessedMenuItem(
                    extracted=item,
                    service_recommendations=service_recs,
                    variation_recommendations=variation_recs,
                    recommendations=combined_recs,
                    selected_service_id=top_service.serviceId if top_service else (top_combined.serviceId if top_combined else None),
                    selected_service_name=top_service.variationName if top_service else canonical_name,
                    selected_variation_id=top_variation.variationId if top_variation else (top_combined.variationId if top_combined else None),
                    selected_variation_name=top_variation.variationName if top_variation else (top_combined.variationName if top_combined else variation_query),
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
