"""Singleton embedding service using SentenceTransformer (BAAI/bge-m3)."""

import logging
from typing import Optional
from sentence_transformers import SentenceTransformer
from service_testing.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    _instance: Optional["EmbeddingService"] = None

    def __init__(self):
        logger.info(
            f"Loading embedding model: {settings.embedding_model_name} on {settings.embedding_device}"
        )
        self.model = SentenceTransformer(
            settings.embedding_model_name,
            device=settings.embedding_device,
        )
        logger.info("Embedding model loaded successfully.")

    @classmethod
    def get_instance(cls) -> "EmbeddingService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def embed_query(self, query: str) -> list[float]:
        """Embeds a single query string into a normalized 1024-dim vector."""
        clean_query = query.strip()
        if not clean_query:
            clean_query = "salon service"
        embedding = self.model.encode(clean_query, normalize_embeddings=True)
        return embedding.tolist()

    def embed_batch(self, queries: list[str]) -> list[list[float]]:
        """Embeds a batch of query strings."""
        clean_queries = [q.strip() if q.strip() else "salon service" for q in queries]
        embeddings = self.model.encode(
            clean_queries,
            batch_size=16,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()
