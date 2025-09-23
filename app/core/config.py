"""Application configuration using Pydantic Settings."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="forbid",
    )

    # API Configuration
    api_title: str = "StackAI Vector Database"
    api_description: str = (
        "REST API for indexing and querying documents in a Vector Database"
    )
    api_version: str = "0.1.0"
    api_prefix: str = "/api/v1"

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    reload: bool = False

    # CORS Configuration
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    cors_allow_credentials: bool = True
    cors_allow_methods: list[str] = Field(default_factory=lambda: ["*"])
    cors_allow_headers: list[str] = Field(default_factory=lambda: ["*"])

    # Cohere API Configuration
    cohere_api_key: str | None = Field(
        default=None, description="Cohere API key for embeddings (optional - uses fake client if not provided)"
    )
    cohere_model: str = "embed-english-v3.0"
    cohere_input_type: str = "search_document"

    # Vector Database Configuration
    default_embedding_dim: int = 1024
    max_chunks_per_library: int = 10000
    max_knn_results: int = 1000
    default_knn_results: int = 10

    # Index Configuration
    default_index_type: str = "linear"  # linear, kdtree, ivf
    index_rebuild_threshold: float = 0.1  # Rebuild when 10% of data changes

    # Logging Configuration
    log_level: str = "INFO"
    log_format_general: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_format_request: str = (
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s | method=%(method)s path=%(path)s status=%(status_code)s duration_ms=%(duration_ms)s request_id=%(request_id)s"
    )


# Global settings instance
settings = Settings()  # type: ignore[call-arg]
