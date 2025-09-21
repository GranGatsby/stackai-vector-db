"""In-memory repository implementations.

This package provides thread-safe, in-memory implementations of the repository
interfaces. These implementations use reader-writer locks to ensure data
consistency during concurrent operations.
"""

from .library_repository import InMemoryLibraryRepository
from .rwlock import RWLock

__all__ = [
    "InMemoryLibraryRepository",
    "RWLock",
]
