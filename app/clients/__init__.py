"""Client modules for external services.

This package contains client implementations for external services
such as embedding providers (Cohere API).
"""

from .embedding import (
    CohereEmbeddingClient,
    EmbeddingClient,
    EmbeddingError,
    FakeEmbeddingClient,
    create_embedding_client,
)

__all__ = [
    "EmbeddingClient",
    "EmbeddingError",
    "FakeEmbeddingClient",
    "CohereEmbeddingClient",
    "create_embedding_client",
]
