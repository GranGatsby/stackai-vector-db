"""Dependency injection providers for API endpoints.

This module contains dependency providers that create and inject services
and repositories into API endpoints. It follows the dependency injection
pattern to decouple the API layer from concrete implementations.
"""

from functools import lru_cache

from app.services import LibraryService
from app.repositories.in_memory import InMemoryLibraryRepository


@lru_cache()
def get_library_repository() -> InMemoryLibraryRepository:
    """Get the library repository instance.

    This function provides a singleton instance of the library repository.
    Using lru_cache ensures the same instance is reused across requests,
    maintaining data consistency in the in-memory implementation.

    Returns:
        The library repository instance
    """
    return InMemoryLibraryRepository()


@lru_cache()
def get_library_service() -> LibraryService:
    """Get the library service instance.

    This function provides a singleton instance of the library service
    with its required dependencies injected. The service is configured
    with the appropriate repository implementation.

    Returns:
        The library service instance
    """
    repository = get_library_repository()
    return LibraryService(repository)
