"""Orchestrator combining Gemini Vision extraction and MongoDB Hybrid RAG variation matching."""

import logging
import time
from typing import Optional

from service_testing.config import settings
from service_testing.schemas import (
    MenuProcessingResponse,
    ProcessedMenuItem,
    VariationRecommendation,
)
from service_testing.services.gemini_extractor import GeminiExtractorService
from service_testing.services.mongo_search import MongoSearchService

logger = logging.getLogger(__name__)


class MenuProcessorService:
    def __init__(
        self,
        gemini_extractor: Optional[GeminiExtractorService] = None,
        mongo_search: Optional[MongoSearchService] = None,
    ):
        self.gemini_extractor = gemini_extractor or GeminiExtractorService()
        self.mongo_search = mongo_search or MongoSearchService()

    def process_menu_image(
        self,
        image_bytes: bytes,
        filename: str,
        mime_type: str = "image/jpeg",
        top_k_recommendations: int = 5,
    ) -> MenuProcessingResponse:
        """
        1. Calls Gemini to extract items from image.
        2. For each extracted variation, performs Atlas Hybrid Search.
        3. Assembles top 5 recommendations with pre-selection.
        """
        start_time = time.time()

        # Step 1: Vision Extraction
        extraction_res = self.gemini_extractor.extract_from_image(
            image_bytes=image_bytes,
            mime_type=mime_type,
        )

        processed_items: list[ProcessedMenuItem] = []

        # Step 2: Hybrid RAG Matching for each item
        for item in extraction_res.items:
            # Query the database with the extracted variation name
            query_text = item.service_variation.strip()
            if not query_text:
                query_text = item.service.replace("_", " ")

            recommendations: list[VariationRecommendation] = []
            try:
                recommendations = self.mongo_search.recommend_variations(
                    query=query_text,
                    top_k=top_k_recommendations,
                )
            except Exception as e:
                logger.error(f"Error retrieving recommendations for '{query_text}': {e}")
                # Provide a graceful fallback candidate
                recommendations = [
                    VariationRecommendation(
                        variationName=query_text,
                        score=1.0,
                        similarity_percentage=90,
                    )
                ]

            top_rec_id = recommendations[0].variationId if recommendations else None
            top_rec_name = recommendations[0].variationName if recommendations else query_text

            processed_items.append(
                ProcessedMenuItem(
                    extracted=item,
                    recommendations=recommendations,
                    selected_variation_id=top_rec_id,
                    selected_variation_name=top_rec_name,
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
