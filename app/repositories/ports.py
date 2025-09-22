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
    """Repository interface for Document entities.

    Defines the contract for document storage operations, enabling
    different implementations while maintaining consistent behavior.
    """

    def list_by_library(
        self, library_id: UUID, limit: int = None, offset: int = 0
    ) -> list[Document]:
        """Retrieve documents in a library with optional pagination.

        Args:
            library_id: The library ID to filter by
            limit: Maximum number of documents to return (None for all)
            offset: Number of documents to skip

        Returns:
            List of Document entities, may be empty
        """
        ...

    def count_by_library(self, library_id: UUID) -> int:
        """Get the total count of documents in a library.

        Args:
            library_id: The library ID to count documents for

        Returns:
            Total number of documents in the library
        """
        ...

    def get_by_id(self, document_id: UUID) -> Document | None:
        """Retrieve a document by its ID.

        Args:
            document_id: The unique identifier of the document

        Returns:
            The Document entity if found, None otherwise
        """
        ...

    def create(self, document: Document) -> Document:
        """Create a new document.

        Args:
            document: The Document entity to create

        Returns:
            The created Document entity

        Raises:
            DocumentAlreadyExistsError: If a document with the same ID exists
        """
        ...

    def update(self, document: Document) -> Document:
        """Update an existing document.

        Args:
            document: The Document entity with updated data

        Returns:
            The updated Document entity

        Raises:
            DocumentNotFoundError: If the document doesn't exist
        """
        ...

    def delete(self, document_id: UUID) -> bool:
        """Delete a document by its ID.

        Args:
            document_id: The unique identifier of the document to delete

        Returns:
            True if the document was deleted, False if it didn't exist
        """
        ...

    def delete_by_library(self, library_id: UUID) -> int:
        """Delete all documents in a library.

        Args:
            library_id: The library ID to delete documents from

        Returns:
            Number of documents deleted
        """
        ...

    def exists(self, document_id: UUID) -> bool:
        """Check if a document exists.

        Args:
            document_id: The unique identifier to check

        Returns:
            True if the document exists, False otherwise
        """
        ...


class ChunkRepository(Protocol):
    """Repository interface for Chunk entities.

    Defines the contract for chunk storage operations, enabling
    different implementations while maintaining consistent behavior.
    """

    def list_by_library(
        self, library_id: UUID, limit: int = None, offset: int = 0
    ) -> list[Chunk]:
        """Retrieve chunks in a library with optional pagination.

        Args:
            library_id: The library ID to filter by
            limit: Maximum number of chunks to return (None for all)
            offset: Number of chunks to skip

        Returns:
            List of Chunk entities, may be empty
        """
        ...

    def list_by_document(
        self, document_id: UUID, limit: int = None, offset: int = 0
    ) -> list[Chunk]:
        """Retrieve chunks in a document with optional pagination.

        Args:
            document_id: The document ID to filter by
            limit: Maximum number of chunks to return (None for all)
            offset: Number of chunks to skip

        Returns:
            List of Chunk entities, may be empty
        """
        ...

    def count_by_library(self, library_id: UUID) -> int:
        """Get the total count of chunks in a library.

        Args:
            library_id: The library ID to count chunks for

        Returns:
            Total number of chunks in the library
        """
        ...

    def count_by_document(self, document_id: UUID) -> int:
        """Get the total count of chunks in a document.

        Args:
            document_id: The document ID to count chunks for

        Returns:
            Total number of chunks in the document
        """
        ...

    def get_by_id(self, chunk_id: UUID) -> Chunk | None:
        """Retrieve a chunk by its ID.

        Args:
            chunk_id: The unique identifier of the chunk

        Returns:
            The Chunk entity if found, None otherwise
        """
        ...

    def create(self, chunk: Chunk) -> Chunk:
        """Create a new chunk.

        Args:
            chunk: The Chunk entity to create

        Returns:
            The created Chunk entity

        Raises:
            ChunkAlreadyExistsError: If a chunk with the same ID exists
        """
        ...

    def update(self, chunk: Chunk) -> Chunk:
        """Update an existing chunk.

        Args:
            chunk: The Chunk entity with updated data

        Returns:
            The updated Chunk entity

        Raises:
            ChunkNotFoundError: If the chunk doesn't exist
        """
        ...

    def delete(self, chunk_id: UUID) -> bool:
        """Delete a chunk by its ID.

        Args:
            chunk_id: The unique identifier of the chunk to delete

        Returns:
            True if the chunk was deleted, False if it didn't exist
        """
        ...

    def delete_by_document(self, document_id: UUID) -> int:
        """Delete all chunks in a document.

        Args:
            document_id: The document ID to delete chunks from

        Returns:
            Number of chunks deleted
        """
        ...

    def delete_by_library(self, library_id: UUID) -> int:
        """Delete all chunks in a library.

        Args:
            library_id: The library ID to delete chunks from

        Returns:
            Number of chunks deleted
        """
        ...

    def exists(self, chunk_id: UUID) -> bool:
        """Check if a chunk exists.

        Args:
            chunk_id: The unique identifier to check

        Returns:
            True if the chunk exists, False otherwise
        """
        ...
