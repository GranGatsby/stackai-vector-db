"""In-memory implementation of the LibraryRepository.

This module provides a thread-safe, in-memory storage implementation for
Library entities using a reader-writer lock to ensure data consistency
during concurrent operations.
"""

from uuid import UUID

from app.domain import Library, LibraryAlreadyExistsError, LibraryNotFoundError
from app.repositories.ports import LibraryRepository

from app.utils import RWLock


class InMemoryLibraryRepository(LibraryRepository):
    """In-memory implementation of LibraryRepository with thread-safe operations.

    This repository stores Library entities in memory using a dictionary
    and provides concurrent access control through a reader-writer lock.
    Multiple readers can access data simultaneously, but writes are exclusive.

    Attributes:
        _libraries: Internal storage mapping UUID to Library entities
        _name_index: Secondary index mapping library names to UUIDs for fast lookups
        _lock: Reader-writer lock for thread-safe operations
    """

    def __init__(self) -> None:
        """Initialize the in-memory repository."""
        self._libraries: dict[UUID, Library] = {}
        self._name_index: dict[str, UUID] = {}
        self._lock = RWLock()

    def _normalize_name(self, name: str) -> str:
        """Normalize library name for case-insensitive comparison.

        Args:
            name: The library name to normalize

        Returns:
            Normalized name for indexing
        """
        return name.casefold()

    def list_all(self, limit: int = None, offset: int = 0) -> list[Library]:
        """Retrieve all libraries with optional pagination.

        Args:
            limit: Maximum number of libraries to return (None for all)
            offset: Number of libraries to skip

        Returns:
            List of Library entities, sorted by name for consistent ordering
        """
        with self._lock.read_lock():
            libraries = list(self._libraries.values())
            sorted_libraries = sorted(libraries, key=lambda lib: lib.name.casefold())

            # Apply pagination
            if offset > 0:
                sorted_libraries = sorted_libraries[offset:]

            if limit is not None:
                sorted_libraries = sorted_libraries[:limit]

            return sorted_libraries

    def count_all(self) -> int:
        """Get the total count of libraries.

        Returns:
            Total number of libraries
        """
        with self._lock.read_lock():
            return len(self._libraries)

    def get_by_id(self, library_id: UUID) -> Library | None:
        """Retrieve a library by its ID.

        Args:
            library_id: The unique identifier of the library

        Returns:
            The Library entity if found, None otherwise
        """
        with self._lock.read_lock():
            return self._libraries.get(library_id)

    def get_by_name(self, name: str) -> Library | None:
        """Retrieve a library by its name.

        Args:
            name: The name of the library

        Returns:
            The Library entity if found, None otherwise
        """
        with self._lock.read_lock():
            normalized_name = self._normalize_name(name)
            library_id = self._name_index.get(normalized_name)
            if library_id is None:
                return None
            return self._libraries.get(library_id)

    def create(self, library: Library) -> Library:
        """Create a new library.

        Args:
            library: The Library entity to create

        Returns:
            The created Library entity

        Raises:
            LibraryAlreadyExistsError: If a library with the same name exists
        """
        with self._lock.write_lock():
            normalized_name = self._normalize_name(library.name)

            # Check for name conflicts
            if normalized_name in self._name_index:
                raise LibraryAlreadyExistsError(library.name)

            # Check for ID conflicts (shouldn't happen with UUIDs, but be safe)
            if library.id in self._libraries:
                raise ValueError(f"Library with ID {library.id} already exists")

            # Store the library and update indexes
            self._libraries[library.id] = library
            self._name_index[normalized_name] = library.id

            return library

    def update(self, library: Library) -> Library:
        """Update an existing library.

        Args:
            library: The Library entity with updated data

        Returns:
            The updated Library entity

        Raises:
            LibraryNotFoundError: If the library doesn't exist
            LibraryAlreadyExistsError: If the new name conflicts with another library
        """
        with self._lock.write_lock():
            # Check if library exists
            if library.id not in self._libraries:
                raise LibraryNotFoundError(str(library.id))

            old_library = self._libraries[library.id]
            old_normalized_name = self._normalize_name(old_library.name)
            new_normalized_name = self._normalize_name(library.name)

            # Check for name conflicts (only if normalized name is actually changing)
            if new_normalized_name != old_normalized_name:
                # Check if the new normalized name is already taken by another library
                existing_library_id = self._name_index.get(new_normalized_name)
                if (
                    existing_library_id is not None
                    and existing_library_id != library.id
                ):
                    raise LibraryAlreadyExistsError(library.name)

                # Update name index
                del self._name_index[old_normalized_name]
                self._name_index[new_normalized_name] = library.id

            # Update the library
            self._libraries[library.id] = library

            return library

    def delete(self, library_id: UUID) -> bool:
        """Delete a library by its ID.

        Args:
            library_id: The unique identifier of the library to delete

        Returns:
            True if the library was deleted, False if it didn't exist
        """
        with self._lock.write_lock():
            library = self._libraries.get(library_id)
            if library is None:
                return False

            # Remove from both indexes
            normalized_name = self._normalize_name(library.name)
            del self._libraries[library_id]
            del self._name_index[normalized_name]

            return True

    def exists(self, library_id: UUID) -> bool:
        """Check if a library exists.

        Args:
            library_id: The unique identifier to check

        Returns:
            True if the library exists, False otherwise
        """
        with self._lock.read_lock():
            return library_id in self._libraries

    def clear(self) -> None:
        """Clear all libraries from the repository.

        This method is primarily useful for testing and cleanup.
        """
        with self._lock.write_lock():
            self._libraries.clear()
            self._name_index.clear()
