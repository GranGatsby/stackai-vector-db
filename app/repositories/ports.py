"""Repository interfaces (ports) for the vector database.

This module defines the repository contracts using Protocol classes,
enabling dependency inversion and allowing different storage implementations
without changing the business logic.
"""

from typing import Protocol
from uuid import UUID

from app.domain import Chunk, Document, Library


class LibraryRepository(Protocol):
    """Repository interface for Library entities.

    Defines the contract for library storage operations, enabling
    different implementations (in-memory, disk, database) while
    maintaining consistent behavior for the business layer.
    """

    def list_all(self, limit: int = None, offset: int = 0) -> list[Library]:
        """Retrieve all libraries with optional pagination.

        Args:
            limit: Maximum number of libraries to return (None for all)
            offset: Number of libraries to skip

        Returns:
            List of Library entities, may be empty
        """
        ...

    def count_all(self) -> int:
        """Get the total count of libraries.

        Returns:
            Total number of libraries
        """
        ...

    def get_by_id(self, library_id: UUID) -> Library | None:
        """Retrieve a library by its ID.

        Args:
            library_id: The unique identifier of the library

        Returns:
            The Library entity if found, None otherwise
        """
        ...

    def get_by_name(self, name: str) -> Library | None:
        """Retrieve a library by its name.

        Args:
            name: The name of the library

        Returns:
            The Library entity if found, None otherwise
        """
        ...

    def create(self, library: Library) -> Library:
        """Create a new library.

        Args:
            library: The Library entity to create

        Returns:
            The created Library entity

        Raises:
            LibraryAlreadyExistsError: If a library with the same name exists
        """
        ...

    def update(self, library: Library) -> Library:
        """Update an existing library.

        Args:
            library: The Library entity with updated data

        Returns:
            The updated Library entity

        Raises:
            LibraryNotFoundError: If the library doesn't exist
        """
        ...

    def delete(self, library_id: UUID) -> bool:
        """Delete a library by its ID.

        Args:
            library_id: The unique identifier of the library to delete

        Returns:
            True if the library was deleted, False if it didn't exist
        """
        ...

    def exists(self, library_id: UUID) -> bool:
        """Check if a library exists.

        Args:
            library_id: The unique identifier to check

        Returns:
            True if the library exists, False otherwise
        """
        ...


class DocumentRepository(Protocol):
    """Repository interface for Document entities."""

    def list_by_library(self, library_id: UUID) -> list[Document]:
        """Retrieve all documents in a library."""
        ...

    def get_by_id(self, document_id: UUID) -> Document | None:
        """Retrieve a document by its ID."""
        ...

    def create(self, document: Document) -> Document:
        """Create a new document."""
        ...

    def update(self, document: Document) -> Document:
        """Update an existing document."""
        ...

    def delete(self, document_id: UUID) -> bool:
        """Delete a document by its ID."""
        ...

    def delete_by_library(self, library_id: UUID) -> int:
        """Delete all documents in a library.

        Returns:
            Number of documents deleted
        """
        ...


class ChunkRepository(Protocol):
    """Repository interface for Chunk entities."""

    def list_by_library(self, library_id: UUID) -> list[Chunk]:
        """Retrieve all chunks in a library."""
        ...

    def list_by_document(self, document_id: UUID) -> list[Chunk]:
        """Retrieve all chunks in a document."""
        ...

    def get_by_id(self, chunk_id: UUID) -> Chunk | None:
        """Retrieve a chunk by its ID."""
        ...

    def create(self, chunk: Chunk) -> Chunk:
        """Create a new chunk."""
        ...

    def update(self, chunk: Chunk) -> Chunk:
        """Update an existing chunk."""
        ...

    def delete(self, chunk_id: UUID) -> bool:
        """Delete a chunk by its ID."""
        ...

    def delete_by_document(self, document_id: UUID) -> int:
        """Delete all chunks in a document.

        Returns:
            Number of chunks deleted
        """
        ...

    def delete_by_library(self, library_id: UUID) -> int:
        """Delete all chunks in a library.

        Returns:
            Number of chunks deleted
        """
        ...
