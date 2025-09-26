"""Document service for business operations."""

# Import for type hints only - will be injected as dependency
from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING

from app.domain import Document, DocumentMetadata, DocumentNotFoundError

if TYPE_CHECKING:
    from uuid import UUID

    from app.repositories.ports import (
        ChunkRepository,
        DocumentRepository,
        LibraryRepository,
    )

if TYPE_CHECKING:
    from app.services.index_service import IndexService


class DocumentService:
    """Service for document business operations."""

    def __init__(
        self,
        document_repository: DocumentRepository,
        library_repository: LibraryRepository,
        chunk_repository: ChunkRepository | None = None,
        index_service: IndexService | None = None,
    ) -> None:
        self._document_repository = document_repository
        self._library_repository = library_repository
        self._chunk_repository = chunk_repository
        self._index_service = index_service

    def list_documents_by_library(
        self, library_id: UUID, limit: int | None = None, offset: int = 0
    ) -> tuple[list[Document], int]:
        if not self._library_repository.exists(library_id):
            raise ValueError(f"Library with ID {library_id} does not exist")

        documents = self._document_repository.list_by_library(
            library_id, limit=limit, offset=offset
        )
        total = self._document_repository.count_by_library(library_id)
        return documents, total

    def get_document(self, document_id: UUID) -> Document:
        document = self._document_repository.get_by_id(document_id)
        if not document:
            raise DocumentNotFoundError(f"Document with ID {document_id} not found")
        return document

    def create_document(
        self,
        library_id: UUID,
        title: str,
        content: str = "",
        metadata: DocumentMetadata | None = None,
    ) -> Document:
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
            with suppress(Exception):
                self._index_service.mark_dirty(library_id)

        return created_document

    def update_document(
        self,
        document_id: UUID,
        title: str | None = None,
        content: str | None = None,
        metadata: DocumentMetadata | None = None,
    ) -> Document:
        existing_document = self.get_document(document_id)

        # Create updated document with new values
        updated_document = existing_document.update(
            title=title, content=content, metadata=metadata
        )

        # Store the updated document
        updated_result = self._document_repository.update(updated_document)

        # Mark index as dirty after document update
        if self._index_service:
            with suppress(Exception):
                self._index_service.mark_dirty(updated_result.library_id)

        return updated_result

    def delete_document(self, document_id: UUID) -> bool:
        """Delete document with cascading deletes: chunks -> document."""
        document = self._document_repository.get_by_id(document_id)
        if not document:
            return False

        # Perform cascading delete of chunks if repository is available
        if self._chunk_repository:
            self._chunk_repository.delete_by_document(document_id)

        # Delete the document itself
        deleted = self._document_repository.delete(document_id)

        # Mark index as dirty after document deletion
        if deleted and self._index_service:
            with suppress(Exception):
                self._index_service.mark_dirty(document.library_id)

        return deleted

    def delete_documents_by_library(self, library_id: UUID) -> int:
        return self._document_repository.delete_by_library(library_id)

    def document_exists(self, document_id: UUID) -> bool:
        return self._document_repository.exists(document_id)

    def count_documents_by_library(self, library_id: UUID) -> int:
        if not self._library_repository.exists(library_id):
            raise ValueError(f"Library with ID {library_id} does not exist")
        return self._document_repository.count_by_library(library_id)
