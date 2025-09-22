"""In-memory implementation of the DocumentRepository.

This module provides a thread-safe, in-memory storage implementation for
Document entities using a reader-writer lock to ensure data consistency
during concurrent operations.
"""

from collections import defaultdict
from uuid import UUID

from app.domain import Document, DocumentNotFoundError
from app.repositories.ports import DocumentRepository

from .rwlock import RWLock


class InMemoryDocumentRepository(DocumentRepository):
    """In-memory implementation of DocumentRepository with thread-safe operations.

    This repository stores Document entities in memory using a dictionary
    and provides concurrent access control through a reader-writer lock.
    Multiple readers can access data simultaneously, but writes are exclusive.

    Attributes:
        _documents: Internal storage mapping UUID to Document entities
        _library_index: Secondary index mapping library_id to set of document UUIDs
        _lock: Reader-writer lock for thread-safe operations
    """

    def __init__(self) -> None:
        """Initialize the in-memory repository."""
        self._documents: dict[UUID, Document] = {}
        self._library_index: dict[UUID, set[UUID]] = defaultdict(set)
        self._lock = RWLock()

    def list_by_library(
        self, library_id: UUID, limit: int = None, offset: int = 0
    ) -> list[Document]:
        """Retrieve documents in a library with optional pagination."""
        with self._lock.read_lock():
            # Get document IDs for this library
            document_ids = list(self._library_index.get(library_id, set()))

            # Sort by title for consistent ordering
            documents = [self._documents[doc_id] for doc_id in document_ids]
            documents.sort(key=lambda d: d.title.lower())

            # Apply pagination
            if offset > 0:
                documents = documents[offset:]
            if limit is not None:
                documents = documents[:limit]

            return documents

    def count_by_library(self, library_id: UUID) -> int:
        """Get the total count of documents in a library."""
        with self._lock.read_lock():
            return len(self._library_index.get(library_id, set()))

    def get_by_id(self, document_id: UUID) -> Document | None:
        """Retrieve a document by its ID."""
        with self._lock.read_lock():
            return self._documents.get(document_id)

    def create(self, document: Document) -> Document:
        """Create a new document."""
        with self._lock.write_lock():
            # Check if document already exists
            if document.id in self._documents:
                raise ValueError(f"Document with ID {document.id} already exists")

            # Store the document
            self._documents[document.id] = document

            # Update secondary index
            self._library_index[document.library_id].add(document.id)

            return document

    def update(self, document: Document) -> Document:
        """Update an existing document."""
        with self._lock.write_lock():
            # Check if document exists
            existing = self._documents.get(document.id)
            if existing is None:
                raise DocumentNotFoundError(f"Document with ID {document.id} not found")

            # Update secondary index if library_id changed
            if existing.library_id != document.library_id:
                self._library_index[existing.library_id].discard(document.id)
                self._library_index[document.library_id].add(document.id)

            # Store updated document
            self._documents[document.id] = document

            return document

    def delete(self, document_id: UUID) -> bool:
        """Delete a document by its ID."""
        with self._lock.write_lock():
            document = self._documents.get(document_id)
            if document is None:
                return False

            # Remove from primary storage
            del self._documents[document_id]

            # Update secondary index
            self._library_index[document.library_id].discard(document_id)

            return True

    def delete_by_library(self, library_id: UUID) -> int:
        """Delete all documents in a library."""
        with self._lock.write_lock():
            document_ids = list(self._library_index.get(library_id, set()))

            # Delete each document (avoiding reentrancy issues)
            deleted_count = 0
            for document_id in document_ids:
                if document_id in self._documents:
                    del self._documents[document_id]
                    deleted_count += 1

            # Clear the library index
            if library_id in self._library_index:
                self._library_index[library_id].clear()

            return deleted_count

    def exists(self, document_id: UUID) -> bool:
        """Check if a document exists."""
        with self._lock.read_lock():
            return document_id in self._documents

    def clear(self) -> None:
        """Clear all documents (for testing purposes)."""
        with self._lock.write_lock():
            self._documents.clear()
            self._library_index.clear()
