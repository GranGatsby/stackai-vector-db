"""Domain layer for the vector database.

This package contains the core business logic, entities, and domain-specific
exceptions. It's framework-agnostic and represents the heart of the application's
business rules.
"""

from .entities import Library, Document, Chunk
from .errors import (
    DomainError,
    LibraryError,
    LibraryNotFoundError,
    LibraryAlreadyExistsError,
    DocumentError,
    DocumentNotFoundError,
    DocumentAlreadyExistsError,
    ChunkError,
    ChunkNotFoundError,
    IndexError,
    IndexNotBuiltError,
    IndexBuildError,
    EmbeddingError,
    EmbeddingDimensionMismatchError,
    ValidationError,
)

__all__ = [
    # Entities
    "Library",
    "Document",
    "Chunk",
    # Errors
    "DomainError",
    "LibraryError",
    "LibraryNotFoundError",
    "LibraryAlreadyExistsError",
    "DocumentError",
    "DocumentNotFoundError",
    "DocumentAlreadyExistsError",
    "ChunkError",
    "ChunkNotFoundError",
    "IndexError",
    "IndexNotBuiltError",
    "IndexBuildError",
    "EmbeddingError",
    "EmbeddingDimensionMismatchError",
    "ValidationError",
]
