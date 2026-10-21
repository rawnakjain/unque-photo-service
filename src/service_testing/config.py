"""Configuration module using pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Vertex AI (Gemini) settings
    gemini_model: str = "gemini-3.5-flash-lite"
    # Tried when the primary model errors out (e.g. outage, retirement). Empty disables.
    gemini_fallback_model: str = "gemini-3.1-flash-lite"
    gcp_project_id: str = "magnetic-alloy-448111-e7"
    # Gemini 3.x models are served from the global endpoint.
    gcp_location: str = "global"
    gemini_max_image_dim: int = 2000
    gemini_thinking_budget: int = 0  # Gemini 2.x models
    gemini_thinking_level: str = "LOW"  # Gemini 3.x+ models
    # "" uses the model default; LOW / MEDIUM / HIGH override it.
    gemini_media_resolution: str = ""
    gemini_max_output_tokens: int = 8192
    gemini_timeout_s: float = 60.0
    gemini_max_attempts: int = 3
    # Send a duplicate request if the first one has streamed nothing after this
    # many seconds (first output normally arrives in ~1-2s). 0 disables.
    gemini_hedge_after_s: float = 3.5

    # MongoDB Atlas settings
    mongodb_uri: str
    mongodb_db_name: str = "Unque"
    mongodb_timeout_ms: int = 5000
    mongodb_min_pool_size: int = 8
    fuzzy_search_workers: int = 16

    # Collections for fuzzy search
    salonservices_collection: str = "salonservices"
    salonservicevariations_collection: str = "salonservicevariations"

    # Text index names for fuzzy search
    salonservices_text_index: str = "salonservices_index"
    salonservicevariations_text_index: str = "salonservicevariations_index"

    # Embedding model settings (kept for potential future use)
    embedding_model_name: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"


settings = Settings()
