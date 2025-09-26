"""Thread-safe in-memory DocumentRepository implementation."""

from collections import defaultdict
from uuid import UUID

from app.domain import Document, DocumentNotFoundError
from app.repositories.ports import DocumentRepository
from app.utils import RWLock


class InMemoryDocumentRepository(DocumentRepository):
    """Thread-safe in-memory document storage."""

    def __init__(self) -> None:
        self._documents: dict[UUID, Document] = {}
        self._library_index: dict[UUID, set[UUID]] = defaultdict(set)
        self._lock = RWLock()

    def list_by_library(
        self, library_id: UUID, limit: int | None = None, offset: int = 0
    ) -> list[Document]:
        with self._lock.read_lock():
            document_ids = self._library_index.get(library_id, set())
            documents = sorted(
                (self._documents[doc_id] for doc_id in document_ids),
                key=lambda d: d.title.lower()
            )
            return documents[offset:offset + limit if limit else None]

    def count_by_library(self, library_id: UUID) -> int:
        with self._lock.read_lock():
            return len(self._library_index.get(library_id, set()))

    def get_by_id(self, document_id: UUID) -> Document | None:
        with self._lock.read_lock():
            return self._documents.get(document_id)

    def create(self, document: Document) -> Document:
        with self._lock.write_lock():
            if document.id in self._documents:
                raise ValueError(f"Document with ID {document.id} already exists")

            self._documents[document.id] = document
            self._library_index[document.library_id].add(document.id)
            return document

    def update(self, document: Document) -> Document:
        with self._lock.write_lock():
            existing = self._documents.get(document.id)
            if not existing:
                raise DocumentNotFoundError(f"Document with ID {document.id} not found")

            if existing.library_id != document.library_id:
                self._library_index[existing.library_id].discard(document.id)
                self._library_index[document.library_id].add(document.id)

            self._documents[document.id] = document
            return document

    def delete(self, document_id: UUID) -> bool:
        with self._lock.write_lock():
            document = self._documents.get(document_id)
            if not document:
                return False

            del self._documents[document_id]
            self._library_index[document.library_id].discard(document_id)
            return True

    def delete_by_library(self, library_id: UUID) -> int:
        with self._lock.write_lock():
            document_ids = list(self._library_index.get(library_id, set()))
            deleted_count = 0
            
            for document_id in document_ids:
                if document_id in self._documents:
                    del self._documents[document_id]
                    deleted_count += 1

            self._library_index[library_id].clear()
            return deleted_count

    def exists(self, document_id: UUID) -> bool:
        with self._lock.read_lock():
            return document_id in self._documents

    def clear(self) -> None:
        with self._lock.write_lock():
            self._documents.clear()
            self._library_index.clear()
