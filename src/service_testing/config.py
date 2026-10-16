"""Configuration module using pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Vertex AI (Gemini) settings
    gemini_model: str = "gemini-2.5-flash"
    gcp_project_id: str = "magnetic-alloy-448111-e7"
    gcp_location: str = "us-central1"

    # MongoDB Atlas settings
    mongodb_uri: str
    mongodb_db_name: str = "Unque"

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
