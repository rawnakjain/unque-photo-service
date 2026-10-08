"""Orchestrator combining Gemini Vision extraction and MongoDB Atlas Fuzzy Search for services & variations."""

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
        1. Calls Gemini to extract items from image.
        2. For each extracted item:
           a. Fuzzy search service in salonservices collection using item.service
           b. Fuzzy search variation in salonservicevariations collection using item.service_variation
        3. Applies 80% confidence threshold.
        4. Returns separate service and variation recommendations.
        """
        start_time = time.time()

        # Step 1: Vision Extraction
        extraction_res = self.gemini_extractor.extract_from_image(
            image_bytes=image_bytes,
            mime_type=mime_type,
        )

        processed_items: list[ProcessedMenuItem] = []

        # Step 2: Fuzzy Search Matching for each item
        for item in extraction_res.items:
            # Step 2a: Search for service in salonservices collection
            service_query = item.service.strip()
            service_recs: list[VariationRecommendation] = []
            if service_query:
                try:
                    service_recs = self.fuzzy_search.recommend_services(
                        query=service_query,
                        top_k=top_k_recommendations,
                        min_confidence=MIN_CONFIDENCE,
                    )
                except Exception as e:
                    logger.error(f"Error searching services for '{service_query}': {e}")

            # Step 2b: Search for variation in salonservicevariations collection
            variation_query = item.service_variation.strip()
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

            # Combined recommendations for backwards compatibility (sorted by similarity)
            all_recommendations = service_recs + variation_recs
            all_recommendations.sort(key=lambda x: x.similarity_percentage or 0, reverse=True)
            combined_recs = all_recommendations[:top_k_recommendations]

            # If no recommendations meet confidence threshold, provide fallback
            if not combined_recs:
                logger.warning(f"No recommendations above {MIN_CONFIDENCE}% confidence for '{variation_query or service_query}'")
                combined_recs = [
                    VariationRecommendation(
                        variationName=variation_query or service_query,
                        score=0.0,
                        similarity_percentage=0,
                    )
                ]

            # Select top service and variation
            top_service = service_recs[0] if service_recs else None
            top_variation = variation_recs[0] if variation_recs else None
            top_combined = combined_recs[0] if combined_recs else None

            processed_items.append(
                ProcessedMenuItem(
                    extracted=item,
                    service_recommendations=service_recs,
                    variation_recommendations=variation_recs,
                    recommendations=combined_recs,  # backwards compat
                    selected_service_id=top_service.serviceId if top_service else None,
                    selected_service_name=top_service.variationName if top_service else service_query,
                    selected_variation_id=top_variation.variationId if top_variation else (top_combined.variationId if top_combined else None),
                    selected_variation_name=top_variation.variationName if top_variation else (top_combined.variationName if top_combined else (variation_query or service_query)),
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
