"""Orchestrator combining Gemini Vision extraction and MongoDB Atlas Fuzzy Search for services and variations."""

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional

from service_testing.config import settings
from service_testing.schemas import (
    ExtractedMenuItem,
    MenuProcessingResponse,
    ProcessedMenuItem,
    ServiceRecommendation,
    VariationRecommendation,
)
from service_testing.services.fuzzy_search import FuzzySearchService
from service_testing.services.gemini_extractor import GeminiExtractorService

logger = logging.getLogger(__name__)

MIN_CONFIDENCE = 80


def _service_lookup_name(item: ExtractedMenuItem) -> Optional[str]:
    return item.canonical_service_name or item.recommended_new_service_name or item.service


def _match_key(item: ExtractedMenuItem) -> tuple[str, str]:
    """Matching depends only on the service lookup name and the variation name."""
    return (
        (_service_lookup_name(item) or "").strip().lower(),
        item.variation_name.strip().lower(),
    )


class MenuProcessorService:
    def __init__(
        self,
        gemini_extractor: Optional[GeminiExtractorService] = None,
        fuzzy_search: Optional[FuzzySearchService] = None,
    ):
        self.gemini_extractor = gemini_extractor or GeminiExtractorService()
        self.fuzzy_search = fuzzy_search or FuzzySearchService()
        self._executor = ThreadPoolExecutor(
            max_workers=settings.fuzzy_search_workers, thread_name_prefix="fuzzy"
        )

    def process_menu_image(
        self,
        image_bytes: bytes,
        filename: str,
        mime_type: str = "image/jpeg",
        top_k_recommendations: int = 5,
    ) -> MenuProcessingResponse:
        """
        1. Streams items from Gemini (includes canonical_service_name from whitelist).
        2. As soon as an item is parsed, its matching starts on a worker thread, so
           matching overlaps with Gemini still generating the rest of the menu.
        3. For each item, the service lookup name uses the priority:
           canonical_service_name > recommended_new_service_name > service
        4. Fuzzy search service in salonservices collection to get serviceId.
        5. Search variations within the matched service when a serviceId is available.
        6. Applies an 80% lexical-similarity threshold to services and variations.
        7. Returns items with serviceId, service name, and variation recommendations (separate lists).
        """
        start_time = time.perf_counter()

        lock = threading.Lock()
        service_cache: dict[str, Future] = {}
        match_futures: dict[tuple[str, str], Future] = {}
        search_ms = {"service": 0.0, "variation": 0.0}

        def recommend_services(service_lookup_name: str) -> list[ServiceRecommendation]:
            # Single-flight per service name: the first caller queries inline and the
            # rest wait on its result. Querying inline (not as a separate executor
            # task) means waiting workers can never starve the query they wait on.
            cache_key = service_lookup_name.strip().lower()
            with lock:
                future = service_cache.get(cache_key)
                is_owner = future is None
                if is_owner:
                    future = service_cache[cache_key] = Future()
            if is_owner:
                service_recs: list[ServiceRecommendation] = []
                _t0 = time.perf_counter()
                try:
                    service_recs = self.fuzzy_search.recommend_services(
                        query=service_lookup_name,
                        top_k=top_k_recommendations,
                        min_confidence=MIN_CONFIDENCE,
                    )
                except Exception:
                    logger.exception("Error searching services for '%s'", service_lookup_name)
                finally:
                    with lock:
                        search_ms["service"] += (time.perf_counter() - _t0) * 1000
                    future.set_result(service_recs)
            return future.result()

        def match(
            item: ExtractedMenuItem,
        ) -> tuple[list[ServiceRecommendation], list[VariationRecommendation]]:
            service_lookup_name = _service_lookup_name(item)
            service_recs = recommend_services(service_lookup_name) if service_lookup_name else []
            top_service = service_recs[0] if service_recs else None
            selected_service_id = top_service.serviceId if top_service else None

            variation_query = item.variation_name.strip()
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
                with lock:
                    search_ms["variation"] += (time.perf_counter() - _t0) * 1000
            return service_recs, variation_recs

        def schedule(item: ExtractedMenuItem) -> None:
            key = _match_key(item)
            with lock:
                if key not in match_futures:
                    match_futures[key] = self._executor.submit(match, item)

        gemini_start = time.perf_counter()
        extraction_res = self.gemini_extractor.extract_from_image(
            image_bytes=image_bytes,
            mime_type=mime_type,
            on_item=schedule,
        )
        gemini_ms = (time.perf_counter() - gemini_start) * 1000

        # Only lookups still running after generation finished add latency.
        fuzzy_tail_start = time.perf_counter()
        processed_items: list[ProcessedMenuItem] = []
        for item in extraction_res.items:
            schedule(item)  # no-op unless the item was not streamed through on_item
            service_recs, variation_recs = match_futures[_match_key(item)].result()
            processed_items.append(self._build_processed_item(item, service_recs, variation_recs))
        fuzzy_phase_ms = (time.perf_counter() - fuzzy_tail_start) * 1000

        duration = time.perf_counter() - start_time
        service_search_ms = search_ms["service"]
        variation_search_ms = search_ms["variation"]
        fuzzy_total_ms = service_search_ms + variation_search_ms

        stage_timings = {
            "gemini_extraction_ms": round(gemini_ms, 1),
            "gemini_first_token_ms": round(extraction_res.first_token_ms or 0.0, 1),
            "fuzzy_phase_wall_ms": round(fuzzy_phase_ms, 1),
            "fuzzy_service_search_ms": round(service_search_ms, 1),
            "fuzzy_variation_search_ms": round(variation_search_ms, 1),
            "fuzzy_search_total_ms": round(fuzzy_total_ms, 1),
            "backend_total_ms": round(duration * 1000, 1),
            "items_processed": len(processed_items),
        }

        logger.info(
            "[stage-timing] gemini=%.1fms first_token=%.1fms fuzzy_phase_wall=%.1fms "
            "fuzzy_services=%.1fms fuzzy_variations=%.1fms fuzzy_total=%.1fms "
            "backend_total=%.1fms items=%d",
            gemini_ms,
            extraction_res.first_token_ms or 0.0,
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
            gemini_model_used=extraction_res.model_used or self.gemini_extractor.model_name,
            stage_timings=stage_timings,
        )

    @staticmethod
    def _build_processed_item(
        item: ExtractedMenuItem,
        service_recs: list[ServiceRecommendation],
        variation_recs: list[VariationRecommendation],
    ) -> ProcessedMenuItem:
        service_lookup_name = _service_lookup_name(item)
        top_service = service_recs[0] if service_recs else None
        variation_query = item.variation_name.strip()
        top_variation = variation_recs[0] if variation_recs else None

        if not service_recs and not variation_recs:
            logger.warning(
                f"No recommendations above {MIN_CONFIDENCE}% confidence for "
                f"service='{service_lookup_name}', variation='{variation_query}'"
            )

        return ProcessedMenuItem(
            extracted=item,
            service_recommendations=service_recs,
            variation_recommendations=variation_recs,
            recommendations=[],  # Deprecated - keep empty
            selected_service_id=top_service.serviceId if top_service else None,
            selected_service_name=top_service.serviceName if top_service else service_lookup_name,
            selected_variation_id=top_variation.variationId if top_variation else None,
            selected_variation_name=top_variation.variationName if top_variation else variation_query,
        )
