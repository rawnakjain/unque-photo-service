"""MongoDB Atlas Fuzzy Search Service using $search with fuzzy option."""

import logging
import re
from typing import Any, List
import pymongo
from pymongo.collection import Collection

from service_testing.config import settings
from service_testing.schemas import VariationRecommendation

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
        Executes MongoDB Atlas Fuzzy Search on salonservices collection.
        Uses $search with fuzzy option to find similar service names.
        """
        coll = self.get_salonservices_collection()

        # Build the fuzzy search pipeline
        # Note: Need $addFields to capture searchScore before filtering in compound queries
        pipeline = [
            {
                "$search": {
                    "index": settings.salonservices_text_index,
                    "compound": {
                        "should": [
                            {
                                "text": {
                                    "query": query,
                                    "path": "salonServiceName",
                                    "fuzzy": {
                                        "maxEdits": 2,
                                        "prefixLength": 0,
                                        "maxExpansions": 50
                                    }
                                }
                            },
                            {
                                "text": {
                                    "query": query,
                                    "path": "serviceNameEnum",
                                    "fuzzy": {
                                        "maxEdits": 2,
                                        "prefixLength": 0,
                                        "maxExpansions": 50
                                    }
                                }
                            }
                        ]
                    }
                }
            },
            {
                "$addFields": {"searchScore": {"$meta": "searchScore"}}
            },
            {
                "$match": {
                    "searchScore": {"$gte": min_score}
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "salonServiceId": 1,
                    "serviceId": 1,
                    "salonServiceName": 1,
                    "serviceNameEnum": 1,
                    "forGender": 1,
                    "score": "$searchScore"
                }
            },
            {
                "$limit": top_k
            },
            {
                "$sort": {"score": -1}
            }
        ]

        try:
            results = list(coll.aggregate(pipeline))
            if not results:
                logger.info(f"Atlas Search returned no results for '{query}' (index may not exist), using regex fallback.")
            else:
                return results
        except Exception as e:
            logger.warning(f"Atlas Search failed for salonservices: {e}. Using regex fallback.")
        
        # Fallback: regex search (works without Atlas Search index, but slower)
        fallback_pipeline = [
            {
                "$match": _build_regex_or_conditions(query, ["salonServiceName", "serviceNameEnum"])
            },
            {
                "$project": {
                    "_id": 1,
                    "salonServiceId": 1,
                    "serviceId": 1,
                    "salonServiceName": 1,
                    "serviceNameEnum": 1,
                    "forGender": 1,
                    "score": {"$literal": 1.0}
                }
            },
            {
                "$limit": top_k
            }
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
    ) -> list[dict[str, Any]]:
        """
        Executes MongoDB Atlas Fuzzy Search on salonservicevariations collection.
        Uses $search with fuzzy option to find similar variation names.
        """
        coll = self.get_salonservicevariations_collection()

        # Build the fuzzy search pipeline
        # Note: Need $addFields to capture searchScore before filtering in compound queries
        pipeline = [
            {
                "$search": {
                    "index": settings.salonservicevariations_text_index,
                    "compound": {
                        "should": [
                            {
                                "text": {
                                    "query": query,
                                    "path": "fullName",
                                    "fuzzy": {
                                        "maxEdits": 2,
                                        "prefixLength": 0,
                                        "maxExpansions": 50
                                    }
                                }
                            },
                            {
                                "text": {
                                    "query": query,
                                    "path": "salonServiceName",
                                    "fuzzy": {
                                        "maxEdits": 2,
                                        "prefixLength": 0,
                                        "maxExpansions": 50
                                    }
                                }
                            },
                            {
                                "text": {
                                    "query": query,
                                    "path": "comment",
                                    "fuzzy": {
                                        "maxEdits": 2,
                                        "prefixLength": 0,
                                        "maxExpansions": 50
                                    }
                                }
                            }
                        ]
                    }
                }
            },
            {
                "$addFields": {"searchScore": {"$meta": "searchScore"}}
            },
            {
                "$match": {
                    "searchScore": {"$gte": min_score}
                }
            },
            {
                "$project": {
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
                    "score": "$searchScore"
                }
            },
            {
                "$limit": top_k
            },
            {
                "$sort": {"score": -1}
            }
        ]

        try:
            results = list(coll.aggregate(pipeline))
            if not results:
                logger.info(f"Atlas Search returned no results for '{query}' (index may not exist), using regex fallback.")
            else:
                return results
        except Exception as e:
            logger.warning(f"Atlas Search failed for salonservicevariations: {e}. Using regex fallback.")
        
        # Fallback: regex search (works without Atlas Search index, but slower)
        fallback_pipeline = [
            {
                "$match": _build_regex_or_conditions(query, ["fullName", "salonServiceName", "comment"])
            },
            {
                "$project": {
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
                }
            },
            {
                "$limit": top_k
            }
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
        Normalizes Atlas Search scores to 0-1 range.
        """
        raw_results = self.fuzzy_search_services(
            query=query,
            top_k=top_k * 3,  # Fetch more to allow for deduplication
            min_score=min_confidence / 100.0
        )
        
        # Normalize scores: Atlas Search BM25 scores can exceed 1.0
        # Use max score in result set as 1.0 (100%) reference
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
            # Normalize to 0-1 range
            normalized_score = min(raw_score / max_score, 1.0)
            similarity_pct = int(round(normalized_score * 100))

            # Only include if meets minimum confidence threshold
            if similarity_pct >= min_confidence:
                recommendations.append(
                    VariationRecommendation(
                        serviceId=service_id,
                        variationId="",  # Not applicable for service-level match
                        variationName=service_name or "Unknown Service",
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
    ) -> list[VariationRecommendation]:
        """
        Returns typed VariationRecommendation models for variations with confidence percentage.
        Deduplicates by variationId (salonServiceVariationId) and by variationName (fullName).
        Normalizes Atlas Search scores to 0-1 range.
        """
        raw_results = self.fuzzy_search_variations(
            query=query,
            top_k=top_k * 3,  # Fetch more to allow for deduplication
            min_score=min_confidence / 100.0
        )
        
        # Normalize scores: Atlas Search BM25 scores can exceed 1.0
        # Use max score in result set as 1.0 (100%) reference
        max_score = max((r.get("score", 0.0) for r in raw_results), default=1.0)
        if max_score == 0:
            max_score = 1.0

        recommendations = []
        seen_variation_ids = set()
        seen_variation_names = set()

        for res in raw_results:
            variation_id = str(res.get("salonServiceVariationId", ""))
            variation_name = res.get("fullName", "").strip()

            # Skip if already seen this variation ID
            if variation_id in seen_variation_ids:
                continue
            # Skip if already seen this variation name (case-insensitive)
            if variation_name.lower() in seen_variation_names:
                continue

            seen_variation_ids.add(variation_id)
            seen_variation_names.add(variation_name.lower())

            raw_score = res.get("score", 0.0)
            # Normalize to 0-1 range
            normalized_score = min(raw_score / max_score, 1.0)
            similarity_pct = int(round(normalized_score * 100))

            # Only include if meets minimum confidence threshold
            if similarity_pct >= min_confidence:
                recommendations.append(
                    VariationRecommendation(
                        serviceId=str(res.get("serviceId", "")),
                        variationId=variation_id,
                        variationName=variation_name or "Unknown Variation",
                        score=normalized_score,
                        similarity_percentage=similarity_pct
                    )
                )

            if len(recommendations) >= top_k:
                break

        return recommendations
