"""MongoDB Atlas Fuzzy Search Service using $search with fuzzy option."""

import logging
import re
from typing import Any, List, Optional
import pymongo
from pymongo.collection import Collection

from service_testing.config import settings
from service_testing.schemas import ServiceRecommendation, VariationRecommendation

logger = logging.getLogger(__name__)


def _build_regex_or_conditions(query: str, fields: list[str]) -> dict:
    """
    Build a MongoDB $or condition for regex search on multiple fields.
    Splits query into words and creates OR conditions for each word on each field.
    Escapes special regex characters.
    """
    words = [w for w in re.split(r"\s+", query.strip()) if w]
    if not words:
        escaped = re.escape(query)
        return {"$or": [{field: {"$regex": escaped, "$options": "i"}} for field in fields]}

    or_conditions = []
    for field in fields:
        for word in words:
            escaped = re.escape(word)
            or_conditions.append({field: {"$regex": escaped, "$options": "i"}})
    return {"$or": or_conditions}

def _is_semantically_valid(query: str, result_name: str) -> bool:
    """
    Strict semantic validator that prevents generic category overlap (e.g., 'Facial')
    from matching unrelated services unless there is genuine token/name continuity.
    """
    q_clean = query.strip().lower()
    r_clean = result_name.strip().lower()

    # Exact or complete phrase containment is always valid
    if q_clean == r_clean or q_clean in r_clean:
        return True

    # Common generic salon terms that shouldn't validate a match on their own
    generic_stop_words = {
        "facial", "massage", "wax", "waxing", "hair", "spa",
        "threading", "cleanup", "bleach", "pack", "mani", "pedi"
    }

    q_words = [w for w in q_clean.split() if len(w) > 1]
    r_words = [w for w in r_clean.split() if len(w) > 1]

    # Filter out pure generic words from the query tokens to find unique intent words
    meaningful_q_words = [w for w in q_words if w not in generic_stop_words]

    # If the query ONLY consisted of generic words (e.g. just "facial"), allow generic matching
    if not meaningful_q_words:
        return any(w in r_words for w in q_words)

    # For queries with specific identifiers (like "Raunak" in "Raunak Facial"),
    # at least one unique/meaningful query word MUST exist inside the result name.
    match_found = False
    for mqw in meaningful_q_words:
        for rw in r_words:
            # Check for strong prefix or exact match of the meaningful word
            if rw == mqw or rw.startswith(mqw) or mqw.startswith(rw):
                match_found = True
                break
        if match_found:
            break

    return match_found

class FuzzySearchService:
    def __init__(self):
        self._client: pymongo.MongoClient | None = None
        self._salonservices: Collection | None = None
        self._salonservicevariations: Collection | None = None

    def get_salonservices_collection(self) -> Collection:
        if self._salonservices is None:
            logger.info(f"Connecting to MongoDB Atlas: {settings.mongodb_db_name}.{settings.salonservices_collection}")
            self._client = pymongo.MongoClient(
                settings.mongodb_uri,
                serverSelectionTimeoutMS=10000,
            )
            self._salonservices = self._client[settings.mongodb_db_name][settings.salonservices_collection]
        return self._salonservices

    def get_salonservicevariations_collection(self) -> Collection:
        if self._salonservicevariations is None:
            logger.info(f"Connecting to MongoDB Atlas: {settings.mongodb_db_name}.{settings.salonservicevariations_collection}")
            if self._client is None:
                self._client = pymongo.MongoClient(
                    settings.mongodb_uri,
                    serverSelectionTimeoutMS=10000,
                )
            self._salonservicevariations = self._client[settings.mongodb_db_name][settings.salonservicevariations_collection]
        return self._salonservicevariations

    def check_connection(self) -> dict[str, Any]:
        """Validates connection and returns database stats."""
        try:
            salonservices = self.get_salonservices_collection()
            variations = self.get_salonservicevariations_collection()
            services_count = salonservices.estimated_document_count()
            variations_count = variations.estimated_document_count()
            return {
                "status": "connected",
                "salonservices_count": services_count,
                "salonservicevariations_count": variations_count
            }
        except Exception as e:
            logger.error(f"MongoDB connection check failed: {e}")
            return {"status": "error", "message": str(e)}

    def fuzzy_search_services(
            self,
            query: str,
            top_k: int = 5,
            min_score: float = 0.8,
    ) -> list[dict[str, Any]]:
        """
        Executes MongoDB Atlas Fuzzy Search on salonservices collection with strict
        exact/prefix matching for short queries (<=4 chars) and phrase-boosting for longer queries.
        """
        coll = self.get_salonservices_collection()
        cleaned_query = query.strip()
        query_len = len(cleaned_query)

        if query_len <= 4:
            # STRICT MODE for short queries: No fuzzy blocks to prevent drift
            search_clauses = [
                {
                    "text": {
                        "query": cleaned_query,
                        "path": "salonServiceName",
                        "score": {"boost": {"value": 5.0}}
                    }
                },
                {
                    "text": {
                        "query": cleaned_query,
                        "path": "serviceNameEnum"
                    }
                }
            ]
        else:
            # FUZZY + PHRASE MODE for longer queries
            search_clauses = [
                {
                    "phrase": {
                        "query": cleaned_query,
                        "path": "salonServiceName",
                        "score": {"boost": {"value": 5.0}}
                    }
                },
                {
                    "text": {
                        "query": cleaned_query,
                        "path": "salonServiceName",
                        "score": {"boost": {"value": 3.0}},
                        "fuzzy": {
                            "maxEdits": 2,
                            "prefixLength": 2
                        }
                    }
                },
                {
                    "text": {
                        "query": cleaned_query,
                        "path": "serviceNameEnum",
                        "fuzzy": {
                            "maxEdits": 2,
                            "prefixLength": 2
                        }
                    }
                }
            ]

        pipeline = [
            {
                "$search": {
                    "index": settings.salonservices_text_index,
                    "compound": {
                        "should": search_clauses,
                        "minimumShouldMatch": 1
                    }
                }
            },
            {"$addFields": {"searchScore": {"$meta": "searchScore"}}},
            {"$match": {"searchScore": {"$gte": min_score}}},
            {"$project": {
                "_id": 1,
                "salonServiceId": 1,
                "serviceId": 1,
                "salonServiceName": 1,
                "serviceNameEnum": 1,
                "forGender": 1,
                "score": "$searchScore"
            }},
            {"$sort": {"score": -1}},
            {"$limit": top_k}
        ]

        try:
            results = list(coll.aggregate(pipeline))
            if results:
                return results
        except Exception as e:
            logger.warning(f"Atlas Search failed for salonservices: {e}. Using regex fallback.")

        # Fallback regex search
        fallback_pipeline = [
            {"$match": _build_regex_or_conditions(cleaned_query, ["salonServiceName", "serviceNameEnum"])},
            {"$project": {
                "_id": 1,
                "salonServiceId": 1,
                "serviceId": 1,
                "salonServiceName": 1,
                "serviceNameEnum": 1,
                "forGender": 1,
                "score": {"$literal": 1.0}
            }},
            {"$limit": top_k}
        ]
        try:
            return list(coll.aggregate(fallback_pipeline))
        except Exception as ex_fallback:
            logger.error(f"Fallback regex search failed: {ex_fallback}")
            return []

    def fuzzy_search_variations(
            self,
            query: str,
            top_k: int = 5,
            min_score: float = 0.8,
            service_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Executes MongoDB Atlas Fuzzy Search on salonservicevariations collection with strict
        exact/prefix matching for short queries (<=4 chars) and phrase-boosting for longer queries.
        Optionally filters by service_id to restrict variations to a specific service.
        """
        coll = self.get_salonservicevariations_collection()
        cleaned_query = query.strip()
        query_len = len(cleaned_query)

        if query_len <= 4:
            # STRICT MODE for short queries: No fuzzy blocks to prevent drift
            search_clauses = [
                {
                    "text": {
                        "query": cleaned_query,
                        "path": "fullName",
                        "score": {"boost": {"value": 5.0}}
                    }
                },
                {
                    "text": {
                        "query": cleaned_query,
                        "path": "salonServiceName"
                    }
                },
                {
                    "text": {
                        "query": cleaned_query,
                        "path": "comment"
                    }
                }
            ]
        else:
            # FUZZY + PHRASE MODE for longer queries
            search_clauses = [
                {
                    "phrase": {
                        "query": cleaned_query,
                        "path": "fullName",
                        "score": {"boost": {"value": 5.0}}
                    }
                },
                {
                    "text": {
                        "query": cleaned_query,
                        "path": "fullName",
                        "score": {"boost": {"value": 3.0}},
                        "fuzzy": {
                            "maxEdits": 2,
                            "prefixLength": 2,
                            "maxExpansions": 50
                        }
                    }
                },
                {
                    "text": {
                        "query": cleaned_query,
                        "path": "salonServiceName",
                        "fuzzy": {
                            "maxEdits": 2,
                            "prefixLength": 2,
                            "maxExpansions": 50
                        }
                    }
                },
                {
                    "text": {
                        "query": cleaned_query,
                        "path": "comment",
                        "fuzzy": {
                            "maxEdits": 2,
                            "prefixLength": 2,
                            "maxExpansions": 50
                        }
                    }
                }
            ]

        # Apply service_id as a regular MongoDB match after Atlas Search.
        # Do not use Atlas Search `equals` here: `serviceId` is not mapped as
        # a `token` field in the current Atlas Search index, which causes errors.
        pipeline = [
            {
                "$search": {
                    "index": settings.salonservicevariations_text_index,
                    "compound": {
                        "should": search_clauses,
                        "minimumShouldMatch": 1,
                    },
                }
            }
        ]

        # Filter before score filtering, projection, and limit so unrelated
        # services cannot crowd out relevant variations.
        if service_id:
            pipeline.append({"$match": {"serviceId": service_id}})

        pipeline.extend([
            {"$addFields": {"searchScore": {"$meta": "searchScore"}}},
            {"$match": {"searchScore": {"$gte": min_score}}},
            {"$project": {
                "_id": 1,
                "salonServiceVariationId": 1,
                "fullName": 1,
                "price": 1,
                "duration": 1,
                "bufferInMins": 1,
                "pricingType": 1,
                "relativeIndex": 1,
                "isCombo": 1,
                "tagNameAndValues": 1,
                "forGender": 1,
                "personType": 1,
                "genericService": 1,
                "salonServiceId": 1,
                "serviceId": 1,
                "salonId": 1,
                "serviceNameEnum": 1,
                "salonServiceName": 1,
                "maxPrice": 1,
                "comment": 1,
                "score": "$searchScore",
            }},
            {"$sort": {"score": -1}},
            {"$limit": top_k * 3},
        ])

        try:
            results = list(coll.aggregate(pipeline))
            if results:
                return results
        except Exception as e:
            logger.warning(
                "Atlas Search failed for variations (service_id=%s): %s. "
                "Using regex fallback.",
                service_id,
                e,
            )

        # Regex fallback (with optional service_id filter)
        fallback_match = _build_regex_or_conditions(cleaned_query, ["fullName", "salonServiceName", "comment"])
        if service_id:
            fallback_match = {"$and": [fallback_match, {"serviceId": service_id}]}
        
        fallback_pipeline = [
            {"$match": fallback_match},
            {"$project": {
                "_id": 1,
                "salonServiceVariationId": 1,
                "fullName": 1,
                "price": 1,
                "duration": 1,
                "bufferInMins": 1,
                "pricingType": 1,
                "relativeIndex": 1,
                "isCombo": 1,
                "tagNameAndValues": 1,
                "forGender": 1,
                "personType": 1,
                "genericService": 1,
                "salonServiceId": 1,
                "serviceId": 1,
                "salonId": 1,
                "serviceNameEnum": 1,
                "salonServiceName": 1,
                "maxPrice": 1,
                "comment": 1,
                "score": {"$literal": 1.0}
            }},
            {"$limit": top_k}
        ]
        try:
            return list(coll.aggregate(fallback_pipeline))
        except Exception as ex_fallback:
            logger.error(f"Fallback regex search failed: {ex_fallback}")
            return []

    def recommend_services(
        self,
        query: str,
        top_k: int = 5,
        min_confidence: int = 80,
    ) -> list[VariationRecommendation]:
        """
        Returns typed VariationRecommendation models for services with confidence percentage.
        Deduplicates by serviceId and by serviceName (salonServiceName).
        Normalizes Atlas Search scores by max score in result set.
        """
        raw_results = self.fuzzy_search_services(
            query=query,
            top_k=top_k * 3,  # Fetch more to allow for deduplication
            min_score=min_confidence / 100.0
        )
        
        # Normalize scores by max score in result set to get relative percentages
        # Atlas Search BM25 scores can exceed 1.0; normalization preserves relative ranking
        max_score = max((r.get("score", 0.0) for r in raw_results), default=1.0)
        if max_score == 0:
            max_score = 1.0

        recommendations = []
        seen_service_ids = set()
        seen_service_names = set()

        for res in raw_results:
            service_id = str(res.get("serviceId", ""))
            service_name = res.get("salonServiceName", "").strip()

            # Skip if already seen this service ID
            if service_id in seen_service_ids:
                continue
            # Skip if already seen this service name (case-insensitive)
            if service_name.lower() in seen_service_names:
                continue

            seen_service_ids.add(service_id)
            seen_service_names.add(service_name.lower())

            raw_score = res.get("score", 0.0)
            # Normalize to 0-1 range by max score in result set
            normalized_score = min(raw_score / max_score, 1.0)
            similarity_pct = int(round(normalized_score * 100))

            # Only include if meets minimum confidence threshold
            if similarity_pct >= min_confidence:
                recommendations.append(
                    ServiceRecommendation(
                        serviceId=service_id,
                        serviceName=service_name or "Unknown Service",
                        score=normalized_score,
                        similarity_percentage=similarity_pct
                    )
                )

            if len(recommendations) >= top_k:
                break

        return recommendations

    def recommend_variations(
            self,
            query: str,
            top_k: int = 5,
            min_confidence: int = 80,
            service_id: Optional[str] = None,
    ) -> list[VariationRecommendation]:
        """
        Returns typed VariationRecommendation models for variations with confidence percentage.
        Deduplicates by variationId and variationName, and filters out semantic drift.
        Optionally filters by service_id to restrict variations to a specific service.
        """
        raw_results = self.fuzzy_search_variations(
            query=query,
            top_k=top_k * 3,  # Fetch more to allow for filtering & deduplication
            min_score=min_confidence / 100.0,  # Filter by Atlas Search score threshold
            service_id=service_id,
        )

        if not raw_results:
            return []

        # Normalize scores by max score in result set to get relative percentages
        # Atlas Search BM25 scores can exceed 1.0; normalization preserves relative ranking
        max_score = max((r.get("score", 0.0) for r in raw_results), default=1.0)
        if max_score == 0:
            max_score = 1.0

        recommendations = []
        seen_variation_ids = set()
        seen_variation_names = set()

        for res in raw_results:
            variation_id = str(res.get("salonServiceVariationId", ""))
            variation_name = res.get("fullName", "").strip()

            # 1. Semantic Firewall: Drop false-positive fuzzy drifts
            if not _is_semantically_valid(query, variation_name):
                continue

            # 2. Skip if already seen this variation ID
            if variation_id in seen_variation_ids:
                continue
            # 3. Skip if already seen this variation name (case-insensitive)
            if variation_name.lower() in seen_variation_names:
                continue

            seen_variation_ids.add(variation_id)
            seen_variation_names.add(variation_name.lower())

            raw_score = res.get("score", 0.0)
            # Normalize to 0-1 range by max score in result set
            normalized_score = min(raw_score / max_score, 1.0)
            similarity_pct = int(round(normalized_score * 100))

            # Only include if meets minimum confidence threshold
            if similarity_pct >= min_confidence:
                recommendations.append(
                    VariationRecommendation(
                        variationId=variation_id,
                        variationName=variation_name or "Unknown Variation",
                        score=normalized_score,
                        similarity_percentage=similarity_pct
                    )
                )

            if len(recommendations) >= top_k:
                break

        return recommendations
