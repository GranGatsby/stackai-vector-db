"""Thread-safe in-memory ChunkRepository implementation."""

from collections import defaultdict
from uuid import UUID

from app.domain import Chunk, ChunkNotFoundError
from app.repositories.ports import ChunkRepository
from app.utils import RWLock


class InMemoryChunkRepository(ChunkRepository):
    """Thread-safe in-memory chunk storage."""

    def __init__(self) -> None:
        self._chunks: dict[UUID, Chunk] = {}
        self._document_index: dict[UUID, set[UUID]] = defaultdict(set)
        self._library_index: dict[UUID, set[UUID]] = defaultdict(set)
        self._lock = RWLock()

    def list_by_library(
        self, library_id: UUID, limit: int | None = None, offset: int = 0
    ) -> list[Chunk]:
        with self._lock.read_lock():
            chunk_ids = self._library_index.get(library_id, set())
            chunks = sorted(
                (self._chunks[chunk_id] for chunk_id in chunk_ids),
                key=lambda c: (c.document_id, c.start_index)
            )
            return chunks[offset:offset + limit if limit else None]

    def list_by_document(
        self, document_id: UUID, limit: int | None = None, offset: int = 0
    ) -> list[Chunk]:
        with self._lock.read_lock():
            chunk_ids = self._document_index.get(document_id, set())
            chunks = sorted(
                (self._chunks[chunk_id] for chunk_id in chunk_ids),
                key=lambda c: c.start_index
            )
            return chunks[offset:offset + limit if limit else None]

    def count_by_library(self, library_id: UUID) -> int:
        """Get the total count of chunks in a library."""
        with self._lock.read_lock():
            return len(self._library_index.get(library_id, set()))

    def count_by_document(self, document_id: UUID) -> int:
        """Get the total count of chunks in a document."""
        with self._lock.read_lock():
            return len(self._document_index.get(document_id, set()))

    def get_by_id(self, chunk_id: UUID) -> Chunk | None:
        """Retrieve a chunk by its ID."""
        with self._lock.read_lock():
            return self._chunks.get(chunk_id)

    def create(self, chunk: Chunk) -> Chunk:
        with self._lock.write_lock():
            # Check if chunk already exists
            if chunk.id in self._chunks:
                raise ValueError(f"Chunk with ID {chunk.id} already exists")

            self._chunks[chunk.id] = chunk
            self._document_index[chunk.document_id].add(chunk.id)
            self._library_index[chunk.library_id].add(chunk.id)
            return chunk

    def update(self, chunk: Chunk) -> Chunk:
        with self._lock.write_lock():
            existing = self._chunks.get(chunk.id)
            if not existing:
                raise ChunkNotFoundError(f"Chunk with ID {chunk.id} not found")

            if existing.document_id != chunk.document_id:
                self._document_index[existing.document_id].discard(chunk.id)
                self._document_index[chunk.document_id].add(chunk.id)

            if existing.library_id != chunk.library_id:
                self._library_index[existing.library_id].discard(chunk.id)
                self._library_index[chunk.library_id].add(chunk.id)

            self._chunks[chunk.id] = chunk
            return chunk

    def delete(self, chunk_id: UUID) -> bool:
        with self._lock.write_lock():
            chunk = self._chunks.get(chunk_id)
            if not chunk:
                return False

            del self._chunks[chunk_id]
            self._document_index[chunk.document_id].discard(chunk_id)
            self._library_index[chunk.library_id].discard(chunk_id)
            return True

    def delete_by_document(self, document_id: UUID) -> int:
        with self._lock.write_lock():
            chunk_ids = list(self._document_index.get(document_id, set()))
            deleted_count = 0

            for chunk_id in chunk_ids:
                chunk = self._chunks.get(chunk_id)
                if chunk:
                    del self._chunks[chunk_id]
                    self._library_index[chunk.library_id].discard(chunk_id)
                    deleted_count += 1

            self._document_index[document_id].clear()
            return deleted_count

    def delete_by_library(self, library_id: UUID) -> int:
        with self._lock.write_lock():
            chunk_ids = list(self._library_index.get(library_id, set()))
            deleted_count = 0

            for chunk_id in chunk_ids:
                chunk = self._chunks.get(chunk_id)
                if chunk:
                    del self._chunks[chunk_id]
                    self._document_index[chunk.document_id].discard(chunk_id)
                    deleted_count += 1

            self._library_index[library_id].clear()
            return deleted_count

    def exists(self, chunk_id: UUID) -> bool:
        with self._lock.read_lock():
            return chunk_id in self._chunks

    def clear(self) -> None:
        with self._lock.write_lock():
            self._chunks.clear()
            self._document_index.clear()
            self._library_index.clear()
