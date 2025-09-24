"""Dependency injection providers for API endpoints.

This module contains dependency providers that create and inject services
and repositories into API endpoints. It follows the dependency injection
pattern to decouple the API layer from concrete implementations.
"""

from functools import lru_cache

from app.clients import EmbeddingClient, create_embedding_client
from app.repositories.in_memory import (
    InMemoryChunkRepository,
    InMemoryDocumentRepository,
    InMemoryLibraryRepository,
)
from app.services import (
    ChunkService,
    DocumentService,
    IndexService,
    LibraryService,
    SearchService,
)


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
    index_service = get_index_service()
    return DocumentService(document_repo, library_repo, chunk_repo, index_service)


@lru_cache
def get_embedding_client() -> EmbeddingClient:
    """Get the embedding client instance.

    This function provides a singleton instance of the embedding client.
    It automatically selects between Cohere API client (if API key is available)
    or fake client (for testing/development).

    Returns:
        The embedding client instance
    """
    return create_embedding_client()


@lru_cache
def get_chunk_service() -> ChunkService:
    """Get the chunk service instance.

    This function provides a singleton instance of the chunk service
    with its required dependencies injected, including the embedding client.

    Returns:
        The chunk service instance
    """
    chunk_repo = get_chunk_repository()
    document_repo = get_document_repository()
    library_repo = get_library_repository()
    embedding_client = get_embedding_client()
    index_service = get_index_service()
    return ChunkService(
        chunk_repo, document_repo, library_repo, embedding_client, index_service
    )


@lru_cache
def get_index_service() -> IndexService:
    """Get the index service instance.

    This function provides a singleton instance of the index service
    with its required dependencies injected.

    Returns:
        The index service instance
    """
    library_repo = get_library_repository()
    chunk_repo = get_chunk_repository()
    embedding_client = get_embedding_client()
    return IndexService(library_repo, chunk_repo, embedding_client)


@lru_cache
def get_search_service() -> SearchService:
    """Get the search service instance.

    This function provides a singleton instance of the search service
    with its required dependencies injected.

    Returns:
        The search service instance
    """
    index_service = get_index_service()
    library_repo = get_library_repository()
    chunk_repo = get_chunk_repository()
    embedding_client = get_embedding_client()
    return SearchService(index_service, library_repo, chunk_repo, embedding_client)
