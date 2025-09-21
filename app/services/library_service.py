"""Library service for orchestrating library-related business operations.

This module contains the LibraryService class that implements the use cases
for library management, providing a clean interface between the API layer
and the domain/repository layers.
"""

from typing import List, Dict, Any, Tuple
from uuid import UUID

from app.domain import Library, LibraryNotFoundError
from app.repositories.ports import LibraryRepository


class LibraryService:
    """Service class for library-related business operations.

    This service orchestrates library use cases, enforcing business rules
    and coordinating between the domain entities and repository layer.
    It provides a clean interface for the API layer while keeping
    business logic separate from HTTP concerns.

    Attributes:
        _repository: The library repository implementation
    """

    def __init__(self, repository: LibraryRepository) -> None:
        """Initialize the library service.

        Args:
            repository: The library repository implementation to use
        """
        self._repository = repository

    def list_libraries(
        self, limit: int = None, offset: int = 0
    ) -> Tuple[List[Library], int]:
        """Retrieve libraries with pagination.

        Args:
            limit: Maximum number of libraries to return (None for all)
            offset: Number of libraries to skip

        Returns:
            Tuple of (libraries list, total count)
        """
        libraries = self._repository.list_all(limit=limit, offset=offset)
        total = self._repository.count_all()
        return libraries, total

    def get_library(self, library_id: UUID) -> Library:
        """Retrieve a library by its ID.

        Args:
            library_id: The unique identifier of the library

        Returns:
            The Library entity

        Raises:
            LibraryNotFoundError: If the library doesn't exist
        """
        library = self._repository.get_by_id(library_id)
        if library is None:
            raise LibraryNotFoundError(str(library_id))
        return library

    def get_library_by_name(self, name: str) -> Library:
        """Retrieve a library by its name.

        Args:
            name: The name of the library

        Returns:
            The Library entity

        Raises:
            LibraryNotFoundError: If the library doesn't exist
        """
        library = self._repository.get_by_name(name)
        if library is None:
            raise LibraryNotFoundError(f"Library with name '{name}' not found")
        return library

    def create_library(
        self, name: str, description: str = "", metadata: Dict[str, Any] = None
    ) -> Library:
        """Create a new library.

        Args:
            name: The library name
            description: Optional description of the library
            metadata: Optional metadata dictionary

        Returns:
            The created Library entity

        Raises:
            LibraryAlreadyExistsError: If a library with the same name exists
            ValueError: If the name is invalid
        """
        # Create the library entity (this validates the name)
        library = Library.create(name=name, description=description, metadata=metadata)

        # Store it in the repository
        return self._repository.create(library)

    def update_library(
        self,
        library_id: UUID,
        name: str = None,
        description: str = None,
        metadata: Dict[str, Any] = None,
    ) -> Library:
        """Update an existing library.

        Args:
            library_id: The unique identifier of the library to update
            name: New name (if provided)
            description: New description (if provided)
            metadata: New metadata (if provided)

        Returns:
            The updated Library entity

        Raises:
            LibraryNotFoundError: If the library doesn't exist
            LibraryAlreadyExistsError: If the new name conflicts with another library
            ValueError: If the new name is invalid
        """
        # Get the existing library
        existing_library = self.get_library(library_id)

        # Create updated library with new values
        updated_library = existing_library.update(
            name=name, description=description, metadata=metadata
        )

        # Store the updated library
        return self._repository.update(updated_library)

    def delete_library(self, library_id: UUID) -> bool:
        """Delete a library by its ID.

        Args:
            library_id: The unique identifier of the library to delete

        Returns:
            True if the library was deleted, False if it didn't exist

        Note:
            In a more complete implementation, this would also handle
            cascading deletes of documents and chunks, possibly through
            domain events or explicit coordination with other services.
        """
        return self._repository.delete(library_id)

    def library_exists(self, library_id: UUID) -> bool:
        """Check if a library exists.

        Args:
            library_id: The unique identifier to check

        Returns:
            True if the library exists, False otherwise
        """
        return self._repository.exists(library_id)

    def library_name_exists(self, name: str) -> bool:
        """Check if a library with the given name exists.

        Args:
            name: The library name to check

        Returns:
            True if a library with that name exists, False otherwise
        """
        library = self._repository.get_by_name(name)
        return library is not None
