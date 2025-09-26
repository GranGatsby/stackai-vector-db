"""Client modules for external services.

This package contains client implementations for external services
such as embedding providers (Cohere API).
"""

from .embedding import (
    CohereEmbeddingClient,
    EmbeddingClient,
    EmbeddingError,
    EmbeddingResult,
    FakeEmbeddingClient,
    create_embedding_client,
)

__all__ = [
    "CohereEmbeddingClient",
    "EmbeddingClient",
    "EmbeddingError",
    "EmbeddingResult",
    "FakeEmbeddingClient",
    "create_embedding_client",
]
