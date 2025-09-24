"""Document service for orchestrating document-related business operations.

This module contains the DocumentService class that implements the use cases
for document management, providing a clean interface between the API layer
and the domain/repository layers.
"""

from uuid import UUID

from app.domain import Document, DocumentMetadata, DocumentNotFoundError
from app.repositories.ports import (
    ChunkRepository,
    DocumentRepository,
    LibraryRepository,
)

# Import for type hints only - will be injected as dependency
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.index_service import IndexService


class DocumentService:
    """Service class for document-related business operations.

    This service orchestrates document use cases, enforcing business rules
    and coordinating between the domain entities and repository layer.
    It provides cross-entity validations and handles cascading operations.

    Attributes:
        _document_repository: The document repository implementation
        _library_repository: The library repository for cross-validation
        _chunk_repository: The chunk repository for cascade operations
    """

    def __init__(
        self,
        document_repository: DocumentRepository,
        library_repository: LibraryRepository,
        chunk_repository: ChunkRepository = None,
        index_service: "IndexService" = None,
    ) -> None:
        """Initialize the document service.

        Args:
            document_repository: The document repository implementation
            library_repository: The library repository for validation
            chunk_repository: The chunk repository for cascade operations (optional)
            index_service: The index service for marking dirty (optional)
        """
        self._document_repository = document_repository
        self._library_repository = library_repository
        self._chunk_repository = chunk_repository
        self._index_service = index_service

    def list_documents_by_library(
        self, library_id: UUID, limit: int = None, offset: int = 0
    ) -> tuple[list[Document], int]:
        """Retrieve documents in a library with pagination.

        Args:
            library_id: The library ID to filter by
            limit: Maximum number of documents to return (None for all)
            offset: Number of documents to skip

        Returns:
            Tuple of (documents list, total count)

        Raises:
            ValueError: If library_id doesn't exist
        """
        # Validate library exists
        if not self._library_repository.exists(library_id):
            raise ValueError(f"Library with ID {library_id} does not exist")

        documents = self._document_repository.list_by_library(
            library_id, limit=limit, offset=offset
        )
        total = self._document_repository.count_by_library(library_id)
        return documents, total

    def get_document(self, document_id: UUID) -> Document:
        """Retrieve a document by its ID.

        Args:
            document_id: The unique identifier of the document

        Returns:
            The Document entity

        Raises:
            DocumentNotFoundError: If the document doesn't exist
        """
        document = self._document_repository.get_by_id(document_id)
        if document is None:
            raise DocumentNotFoundError(f"Document with ID {document_id} not found")
        return document

    def create_document(
        self,
        library_id: UUID,
        title: str,
        content: str = "",
        metadata: DocumentMetadata | None = None,
    ) -> Document:
        """Create a new document.

        Args:
            library_id: The library ID where the document will be created
            title: The document title
            content: The document content
            metadata: Additional document metadata

        Returns:
            The created Document entity

        Raises:
            ValueError: If library_id doesn't exist or title is invalid
        """
        # Validate library exists
        if not self._library_repository.exists(library_id):
            raise ValueError(f"Library with ID {library_id} does not exist")

        # Create the document using domain factory
        document = Document.create(
            library_id=library_id,
            title=title,
            content=content,
            metadata=metadata,
        )

        # Store the document
        created_document = self._document_repository.create(document)
        
        # Mark index as dirty after document creation
        if self._index_service:
            try:
                self._index_service.mark_dirty(library_id)
            except Exception:
                # Don't fail document creation if index marking fails
                pass
        
        return created_document

    def update_document(
        self,
        document_id: UUID,
        title: str = None,
        content: str = None,
        metadata: DocumentMetadata | None = None,
    ) -> Document:
        """Update an existing document with partial data.

        Args:
            document_id: The unique identifier of the document to update
            title: New title (if provided)
            content: New content (if provided)
            metadata: New metadata (if provided)

        Returns:
            The updated Document entity

        Raises:
            DocumentNotFoundError: If the document doesn't exist
        """
        # Get the existing document
        existing_document = self.get_document(document_id)

        # Create updated document with new values
        updated_document = existing_document.update(
            title=title, content=content, metadata=metadata
        )

        # Store the updated document
        updated_result = self._document_repository.update(updated_document)
        
        # Mark index as dirty after document update
        if self._index_service:
            try:
                self._index_service.mark_dirty(updated_result.library_id)
            except Exception:
                # Don't fail document update if index marking fails
                pass
        
        return updated_result

    def delete_document(self, document_id: UUID) -> bool:
        """Delete a document by its ID with cascading deletes.

        This method performs cascading deletes:
        1. Delete all chunks in the document
        2. Delete the document itself

        Args:
            document_id: The unique identifier of the document to delete

        Returns:
            True if the document was deleted, False if it didn't exist
        """
        # Check if document exists and get library_id for dirty marking
        document = self._document_repository.get_by_id(document_id)
        if document is None:
            return False

        # Perform cascading delete of chunks if repository is available
        if self._chunk_repository:
            chunks_deleted = self._chunk_repository.delete_by_document(document_id)

        # Delete the document itself
        deleted = self._document_repository.delete(document_id)
        
        # Mark index as dirty after document deletion
        if deleted and self._index_service:
            try:
                self._index_service.mark_dirty(document.library_id)
            except Exception:
                # Don't fail document deletion if index marking fails
                pass
        
        return deleted

    def delete_documents_by_library(self, library_id: UUID) -> int:
        """Delete all documents in a library (cascade operation).

        Args:
            library_id: The library ID to delete documents from

        Returns:
            Number of documents deleted
        """
        return self._document_repository.delete_by_library(library_id)

    def document_exists(self, document_id: UUID) -> bool:
        """Check if a document exists.

        Args:
            document_id: The unique identifier to check

        Returns:
            True if the document exists, False otherwise
        """
        return self._document_repository.exists(document_id)

    def count_documents_by_library(self, library_id: UUID) -> int:
        """Get the count of documents in a library.

        Args:
            library_id: The library ID to count documents for

        Returns:
            Number of documents in the library

        Raises:
            ValueError: If library_id doesn't exist
        """
        # Validate library exists
        if not self._library_repository.exists(library_id):
            raise ValueError(f"Library with ID {library_id} does not exist")

        return self._document_repository.count_by_library(library_id)
