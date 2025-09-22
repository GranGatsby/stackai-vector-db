"""Domain layer for the vector database.

This package contains the core business logic, entities, and domain-specific
exceptions. It's framework-agnostic and represents the heart of the application's
business rules.
"""

from .entities import Chunk, ChunkMetadata, Document, DocumentMetadata, Library, LibraryMetadata
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
    "LibraryMetadata",
    "Document",
    "DocumentMetadata",
    "Chunk",
    "ChunkMetadata",
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
