"""Domain layer for the vector database.

This package contains the core business logic, entities, and domain-specific
exceptions. It's framework-agnostic and represents the heart of the application's
business rules.
"""

from .entities import (
    Chunk,
    ChunkMetadata,
    Document,
    DocumentMetadata,
    Library,
    LibraryMetadata,
)
from .errors import (
    ChunkError,
    ChunkNotFoundError,
    DocumentAlreadyExistsError,
    DocumentError,
    DocumentNotFoundError,
    DomainError,
    EmbeddingDimensionMismatchError,
    EmbeddingError,
    EmptyLibraryError,
    InvalidSearchParameterError,
    LibraryAlreadyExistsError,
    LibraryError,
    LibraryNotFoundError,
    SearchError,
    ValidationError,
    VectorIndexBuildError,
    VectorIndexError,
    VectorIndexNotBuiltError,
)

__all__ = [
    # Entities
    "Chunk",
    "ChunkMetadata",
    "Document",
    "DocumentMetadata",
    "Library",
    "LibraryMetadata",
    # Errors
    "ChunkError",
    "ChunkNotFoundError",
    "DocumentAlreadyExistsError",
    "DocumentError",
    "DocumentNotFoundError",
    "DomainError",
    "EmbeddingDimensionMismatchError",
    "EmbeddingError",
    "EmptyLibraryError",
    "InvalidSearchParameterError",
    "LibraryAlreadyExistsError",
    "LibraryError",
    "LibraryNotFoundError",
    "SearchError",
    "ValidationError",
    "VectorIndexBuildError",
    "VectorIndexError",
    "VectorIndexNotBuiltError",
]
