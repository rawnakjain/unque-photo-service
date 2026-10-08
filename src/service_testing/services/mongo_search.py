"""MongoDB Atlas Hybrid Search Service using $rankFusion."""

import logging
from typing import Any
import pymongo
from pymongo.collection import Collection

from service_testing.config import settings
from service_testing.schemas import VariationRecommendation
from service_testing.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class MongoSearchService:
    def __init__(self):
        self._client: pymongo.MongoClient | None = None
        self._collection: Collection | None = None
        self.embedding_service = EmbeddingService.get_instance()

    def get_collection(self) -> Collection:
        if self._collection is None:
            logger.info(f"Connecting to MongoDB Atlas: {settings.mongodb_db_name}.{settings.mongodb_collection}")
            self._client = pymongo.MongoClient(
                settings.mongodb_uri,
                serverSelectionTimeoutMS=10000,
            )
            self._collection = self._client[settings.mongodb_db_name][settings.mongodb_collection]
        return self._collection

    def check_connection(self) -> dict[str, Any]:
        """Validates connection and returns database stats."""
        try:
            coll = self.get_collection()
            count = coll.estimated_document_count()
            return {"status": "connected", "document_count": count}
        except Exception as e:
            logger.error(f"MongoDB connection check failed: {e}")
            return {"status": "error", "message": str(e)}

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        candidate_limit: int = 50,
        vector_weight: float = 0.5,
        text_weight: float = 0.5,
    ) -> list[dict[str, Any]]:
        """
        Executes MongoDB Atlas Hybrid Search ($rankFusion)
        combining vector search via BAAI/bge-m3 embeddings and BM25 full-text search.
        """
        coll = self.get_collection()
        query_vector = self.embedding_service.embed_query(query)

        pipeline = [
            {
                "$rankFusion": {
                    "input": {
                        "pipelines": {
                            # 1. Vector Search
                            "vectorPipeline": [
                                {
                                    "$vectorSearch": {
                                        "index": settings.mongodb_vector_index,
                                        "path": "embedding",
                                        "queryVector": query_vector,
                                        "numCandidates": 500,
                                        "limit": candidate_limit,
                                    }
                                }
                            ],
                            # 2. BM25 Full Text Search
                            "fullTextPipeline": [
                                {
                                    "$search": {
                                        "index": settings.mongodb_text_index,
                                        "text": {
                                            "query": query,
                                            "path": "variationName",
                                        },
                                    }
                                },
                                {
                                    "$limit": candidate_limit,
                                },
                            ],
                        }
                    },
                    "combination": {
                        "weights": {
                            "vectorPipeline": vector_weight,
                            "fullTextPipeline": text_weight,
                        }
                    },
                    "scoreDetails": True,
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "serviceId": 1,
                    "variationId": 1,
                    "variationName": 1,
                    "score": {"$meta": "score"},
                    "scoreDetails": {"$meta": "scoreDetails"},
                }
            },
            {
                "$limit": top_k,
            },
        ]

        try:
            results = list(coll.aggregate(pipeline))
            return results
        except Exception as e:
            logger.warning(f"Hybrid search with rankFusion failed: {e}. Falling back to vector search only.")
            # Fallback directly to vector search if fullText index is busy or single pipeline desired
            fallback_pipeline = [
                {
                    "$vectorSearch": {
                        "index": settings.mongodb_vector_index,
                        "path": "embedding",
                        "queryVector": query_vector,
                        "numCandidates": 100,
                        "limit": top_k,
                    }
                },
                {
                    "$project": {
                        "_id": 1,
                        "serviceId": 1,
                        "variationId": 1,
                        "variationName": 1,
                        "score": {"$meta": "vectorSearchScore"},
                    }
                },
            ]
            try:
                return list(coll.aggregate(fallback_pipeline))
            except Exception as ex_vec:
                logger.error(f"Fallback vector search also failed: {ex_vec}")
                raise ex_vec

    def recommend_variations(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[VariationRecommendation]:
        """Returns typed VariationRecommendation models with normalized similarity percentage."""
        raw_results = self.hybrid_search(query=query, top_k=top_k)
        recommendations = []

        max_score = raw_results[0].get("score", 0.02) if raw_results else 0.02

        for res in raw_results:
            raw_score = res.get("score")
            # Calculate a pleasant visual similarity percentage
            if raw_score is not None:
                if raw_score <= 1.0 and raw_score > 0.05:
                    pct = int(round(raw_score * 100))
                else:
                    # RRF reciprocal rank fusion score (e.g. ~0.016 -> ~95% relative scale)
                    pct = min(99, max(50, int(round((raw_score / (max_score if max_score > 0 else 0.02)) * 95))))
            else:
                pct = 75

            recommendations.append(
                VariationRecommendation(
                    serviceId=str(res.get("serviceId", "")),
                    variationId=str(res.get("variationId", "")),
                    variationName=res.get("variationName", "Unknown Variation"),
                    score=raw_score,
                    similarity_percentage=pct,
                )
            )

        return recommendations
