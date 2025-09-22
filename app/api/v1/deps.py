"""Dependency injection providers for API endpoints.

This module contains dependency providers that create and inject services
and repositories into API endpoints. It follows the dependency injection
pattern to decouple the API layer from concrete implementations.
"""

from functools import lru_cache

from app.repositories.in_memory import (
    InMemoryChunkRepository,
    InMemoryDocumentRepository,
    InMemoryLibraryRepository,
)
from app.services import ChunkService, DocumentService, LibraryService


@lru_cache
def get_library_repository() -> InMemoryLibraryRepository:
    """Get the library repository instance.

    This function provides a singleton instance of the library repository.
    Using lru_cache ensures the same instance is reused across requests,
    maintaining data consistency in the in-memory implementation.

    Returns:
        The library repository instance
    """
    return InMemoryLibraryRepository()


@lru_cache
def get_document_repository() -> InMemoryDocumentRepository:
    """Get the document repository instance.

    Returns:
        The document repository instance
    """
    return InMemoryDocumentRepository()


@lru_cache
def get_chunk_repository() -> InMemoryChunkRepository:
    """Get the chunk repository instance.

    Returns:
        The chunk repository instance
    """
    return InMemoryChunkRepository()


@lru_cache
def get_library_service() -> LibraryService:
    """Get the library service instance with cascade support.

    This function provides a singleton instance of the library service
    with its required dependencies injected, including repositories
    for cascading delete operations.

    Returns:
        The library service instance
    """
    library_repo = get_library_repository()
    document_repo = get_document_repository()
    chunk_repo = get_chunk_repository()
    return LibraryService(library_repo, document_repo, chunk_repo)


@lru_cache
def get_document_service() -> DocumentService:
    """Get the document service instance.

    This function provides a singleton instance of the document service
    with its required dependencies injected.

    Returns:
        The document service instance
    """
    document_repo = get_document_repository()
    library_repo = get_library_repository()
    chunk_repo = get_chunk_repository()
    return DocumentService(document_repo, library_repo, chunk_repo)


@lru_cache
def get_chunk_service() -> ChunkService:
    """Get the chunk service instance.

    This function provides a singleton instance of the chunk service
    with its required dependencies injected.

    Returns:
        The chunk service instance
    """
    chunk_repo = get_chunk_repository()
    document_repo = get_document_repository()
    library_repo = get_library_repository()
    return ChunkService(chunk_repo, document_repo, library_repo)
