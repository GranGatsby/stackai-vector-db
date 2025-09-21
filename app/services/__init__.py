"""Service layer for the vector database.

This package contains the application services that orchestrate business
operations and use cases. Services coordinate between the domain layer
and repository layer while keeping business logic separate from API concerns.
"""

from .library_service import LibraryService

__all__ = [
    "LibraryService",
]
