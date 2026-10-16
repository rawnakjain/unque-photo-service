"""Orchestrator combining Gemini Vision extraction and MongoDB Atlas Fuzzy Search for services and variations."""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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
MAX_FUZZY_WORKERS = 8


class MenuProcessorService:
    def __init__(
        self,
        gemini_extractor: Optional[GeminiExtractorService] = None,
        fuzzy_search: Optional[FuzzySearchService] = None,
    ):
        self.gemini_extractor = gemini_extractor or GeminiExtractorService()
        self.fuzzy_search = fuzzy_search or FuzzySearchService()
        self._executor = ThreadPoolExecutor(
            max_workers=MAX_FUZZY_WORKERS, thread_name_prefix="fuzzy"
        )

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
        4. Search variations within the matched service when a serviceId is available.
        5. Applies an 80% lexical-similarity threshold to services and variations.
        6. Returns items with serviceId, service name, and variation recommendations (separate lists).
        """
        start_time = time.perf_counter()

        gemini_start = time.perf_counter()
        extraction_res = self.gemini_extractor.extract_from_image(
            image_bytes=image_bytes,
            mime_type=mime_type,
        )
        gemini_ms = (time.perf_counter() - gemini_start) * 1000

        processed_items: list[ProcessedMenuItem] = []
        service_search_ms = 0.0
        variation_search_ms = 0.0
        fuzzy_phase_ms = 0.0

        items = extraction_res.items

        if items:
            # Warm up the Mongo client / Atlas Search before parallel queries.
            self.fuzzy_search.check_connection()

            service_cache: dict[str, list[ServiceRecommendation]] = {}
            cache_lock = threading.Lock()

            def process_one(item):
                service_lookup_name = (
                    item.canonical_service_name
                    or item.recommended_new_service_name
                    or item.service
                )

                s_ms = 0.0
                service_recs: list[ServiceRecommendation] = []
                if service_lookup_name:
                    cache_key = service_lookup_name.strip().lower()
                    with cache_lock:
                        cache_hit = cache_key in service_cache
                        service_recs = service_cache.get(cache_key, [])
                    if not cache_hit:
                        _t0 = time.perf_counter()
                        try:
                            service_recs = self.fuzzy_search.recommend_services(
                                query=service_lookup_name,
                                top_k=top_k_recommendations,
                                min_confidence=MIN_CONFIDENCE,
                            )
                        except Exception:
                            logger.exception("Error searching services for '%s'", service_lookup_name)
                        s_ms = (time.perf_counter() - _t0) * 1000
                        with cache_lock:
                            service_cache[cache_key] = service_recs

                top_service = service_recs[0] if service_recs else None
                selected_service_id = top_service.serviceId if top_service else None
                selected_service_name = top_service.serviceName if top_service else service_lookup_name

                variation_query = item.variation_name.strip()
                v_ms = 0.0
                variation_recs: list[VariationRecommendation] = []
                if variation_query:
                    _t0 = time.perf_counter()
                    try:
                        variation_recs = self.fuzzy_search.recommend_variations(
                            query=variation_query,
                            top_k=top_k_recommendations,
                            min_confidence=MIN_CONFIDENCE,
                            service_id=selected_service_id,
                        )
                    except Exception:
                        logger.exception(
                            "Error searching variations for '%s' (service_id=%s)",
                            variation_query,
                            selected_service_id,
                        )
                    v_ms = (time.perf_counter() - _t0) * 1000

                top_variation = variation_recs[0] if variation_recs else None
                selected_variation_id = top_variation.variationId if top_variation else None
                selected_variation_name = top_variation.variationName if top_variation else variation_query

                if not service_recs and not variation_recs:
                    logger.warning(
                        f"No recommendations above {MIN_CONFIDENCE}% confidence for "
                        f"service='{service_lookup_name}', variation='{variation_query}'"
                    )

                processed = ProcessedMenuItem(
                    extracted=item,
                    service_recommendations=service_recs,
                    variation_recommendations=variation_recs,
                    recommendations=[],  # Deprecated - keep empty
                    selected_service_id=selected_service_id,
                    selected_service_name=selected_service_name,
                    selected_variation_id=selected_variation_id,
                    selected_variation_name=selected_variation_name,
                )
                return processed, s_ms, v_ms

            fuzzy_phase_start = time.perf_counter()
            results = list(self._executor.map(process_one, items))
            fuzzy_phase_ms = (time.perf_counter() - fuzzy_phase_start) * 1000

            for processed, s_ms, v_ms in results:
                processed_items.append(processed)
                service_search_ms += s_ms
                variation_search_ms += v_ms

        duration = time.perf_counter() - start_time
        fuzzy_total_ms = service_search_ms + variation_search_ms

        stage_timings = {
            "gemini_extraction_ms": round(gemini_ms, 1),
            "fuzzy_phase_wall_ms": round(fuzzy_phase_ms, 1),
            "fuzzy_service_search_ms": round(service_search_ms, 1),
            "fuzzy_variation_search_ms": round(variation_search_ms, 1),
            "fuzzy_search_total_ms": round(fuzzy_total_ms, 1),
            "backend_total_ms": round(duration * 1000, 1),
            "items_processed": len(processed_items),
        }

        logger.info(
            "[stage-timing] gemini=%.1fms fuzzy_phase_wall=%.1fms fuzzy_services=%.1fms "
            "fuzzy_variations=%.1fms fuzzy_total=%.1fms backend_total=%.1fms items=%d",
            gemini_ms,
            fuzzy_phase_ms,
            service_search_ms,
            variation_search_ms,
            fuzzy_total_ms,
            duration * 1000,
            len(processed_items),
        )

        return MenuProcessingResponse(
            image_filename=filename,
            total_items=len(processed_items),
            items=processed_items,
            processing_time_seconds=round(duration, 2),
            gemini_model_used=self.gemini_extractor.model_name,
            stage_timings=stage_timings,
        )
