"""Domain layer for the vector database.

This package contains the core business logic, entities, and domain-specific
exceptions. It's framework-agnostic and represents the heart of the application's
business rules.
"""

from .entities import Chunk, Document, Library
from .errors import (
    ChunkError,
    ChunkNotFoundError,
    DocumentAlreadyExistsError,
    DocumentError,
    DocumentNotFoundError,
    DomainError,
    EmbeddingDimensionMismatchError,
    EmbeddingError,
    IndexBuildError,
    IndexError,
    IndexNotBuiltError,
    LibraryAlreadyExistsError,
    LibraryError,
    LibraryNotFoundError,
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
