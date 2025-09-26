"""Repository interfaces for dependency inversion."""

from typing import Protocol
from uuid import UUID

from app.domain import Chunk, Document, Library


class LibraryRepository(Protocol):
    """Repository interface for Library entities."""

    def list_all(self, limit: int | None = None, offset: int = 0) -> list[Library]:
        """Retrieve all libraries with optional pagination."""
        ...

    def count_all(self) -> int:
        """Get the total count of libraries"""
        ...

    def get_by_id(self, library_id: UUID) -> Library | None:
        """Retrieve a library by its ID"""
        ...

    def get_by_name(self, name: str) -> Library | None:
        """Retrieve a library by its name"""
        ...

    def create(self, library: Library) -> Library:
        """Create a new library"""
        ...

    def update(self, library: Library) -> Library:
        """Update an existing library"""
        ...

    def delete(self, library_id: UUID) -> bool:
        """Delete a library by its ID"""
        ...

    def exists(self, library_id: UUID) -> bool:
        """Check if a library exists"""
        ...


class DocumentRepository(Protocol):
    """Repository interface for Document entities"""

    def list_by_library(
        self, library_id: UUID, limit: int | None = None, offset: int = 0
    ) -> list[Document]:
        """Retrieve documents in a library with optional pagination"""
        ...

    def count_by_library(self, library_id: UUID) -> int:
        """Get the total count of documents in a library"""
        ...

    def get_by_id(self, document_id: UUID) -> Document | None:
        """Retrieve a document by its ID"""
        ...

    def create(self, document: Document) -> Document:
        """Create a new document"""
        ...

    def update(self, document: Document) -> Document:
        """Update an existing document"""
        ...

    def delete(self, document_id: UUID) -> bool:
        """Delete a document by its ID"""
        ...

    def delete_by_library(self, library_id: UUID) -> int:
        """Delete all documents in a library"""
        ...

    def exists(self, document_id: UUID) -> bool:
        """Check if a document exists"""
        ...


class ChunkRepository(Protocol):
    """Repository interface for Chunk entities."""

    def list_by_library(
        self, library_id: UUID, limit: int | None = None, offset: int = 0
    ) -> list[Chunk]:
        """Retrieve chunks in a library with optional pagination"""
        ...

    def list_by_document(
        self, document_id: UUID, limit: int | None = None, offset: int = 0
    ) -> list[Chunk]:
        """Retrieve chunks in a document with optional pagination"""
        ...

    def count_by_library(self, library_id: UUID) -> int:
        """Get the total count of chunks in a library"""
        ...

    def count_by_document(self, document_id: UUID) -> int:
        """Get the total count of chunks in a document"""
        ...

    def get_by_id(self, chunk_id: UUID) -> Chunk | None:
        """Retrieve a chunk by its ID"""
        ...

    def create(self, chunk: Chunk) -> Chunk:
        """Create a new chunk"""
        ...

    def update(self, chunk: Chunk) -> Chunk:
        """Update an existing chunk"""
        ...

    def delete(self, chunk_id: UUID) -> bool:
        """Delete a chunk by its ID."""
        ...

    def delete_by_document(self, document_id: UUID) -> int:
        """Delete all chunks in a document."""
        ...

    def delete_by_library(self, library_id: UUID) -> int:
        """Delete all chunks in a library."""
        ...

    def exists(self, chunk_id: UUID) -> bool:
        """Check if a chunk exists"""
        ...
