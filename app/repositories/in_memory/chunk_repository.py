"""In-memory implementation of the ChunkRepository.

This module provides a thread-safe, in-memory storage implementation for
Chunk entities using a reader-writer lock to ensure data consistency
during concurrent operations.
"""

from collections import defaultdict
from uuid import UUID

from app.domain import Chunk, ChunkNotFoundError
from app.repositories.ports import ChunkRepository

from app.utils import RWLock


class InMemoryChunkRepository(ChunkRepository):
    """In-memory implementation of ChunkRepository with thread-safe operations.

    This repository stores Chunk entities in memory using a dictionary
    and provides concurrent access control through a reader-writer lock.
    Multiple readers can access data simultaneously, but writes are exclusive.

    Attributes:
        _chunks: Internal storage mapping UUID to Chunk entities
        _document_index: Secondary index mapping document_id to set of chunk UUIDs
        _library_index: Secondary index mapping library_id to set of chunk UUIDs
        _lock: Reader-writer lock for thread-safe operations
    """

    def __init__(self) -> None:
        """Initialize the in-memory repository."""
        self._chunks: dict[UUID, Chunk] = {}
        self._document_index: dict[UUID, set[UUID]] = defaultdict(set)
        self._library_index: dict[UUID, set[UUID]] = defaultdict(set)
        self._lock = RWLock()

    def list_by_library(
        self, library_id: UUID, limit: int = None, offset: int = 0
    ) -> list[Chunk]:
        """Retrieve chunks in a library with optional pagination."""
        with self._lock.read_lock():
            # Get chunk IDs for this library
            chunk_ids = list(self._library_index.get(library_id, set()))

            # Sort by start_index for consistent ordering
            chunks = [self._chunks[chunk_id] for chunk_id in chunk_ids]
            chunks.sort(key=lambda c: (c.document_id, c.start_index))

            # Apply pagination
            if offset > 0:
                chunks = chunks[offset:]
            if limit is not None:
                chunks = chunks[:limit]

            return chunks

    def list_by_document(
        self, document_id: UUID, limit: int = None, offset: int = 0
    ) -> list[Chunk]:
        """Retrieve chunks in a document with optional pagination."""
        with self._lock.read_lock():
            # Get chunk IDs for this document
            chunk_ids = list(self._document_index.get(document_id, set()))

            # Sort by start_index for consistent ordering
            chunks = [self._chunks[chunk_id] for chunk_id in chunk_ids]
            chunks.sort(key=lambda c: c.start_index)

            # Apply pagination
            if offset > 0:
                chunks = chunks[offset:]
            if limit is not None:
                chunks = chunks[:limit]

            return chunks

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
        """Create a new chunk."""
        with self._lock.write_lock():
            # Check if chunk already exists
            if chunk.id in self._chunks:
                raise ValueError(f"Chunk with ID {chunk.id} already exists")

            # Store the chunk
            self._chunks[chunk.id] = chunk

            # Update secondary indexes
            self._document_index[chunk.document_id].add(chunk.id)
            self._library_index[chunk.library_id].add(chunk.id)

            return chunk

    def update(self, chunk: Chunk) -> Chunk:
        """Update an existing chunk."""
        with self._lock.write_lock():
            # Check if chunk exists
            existing = self._chunks.get(chunk.id)
            if existing is None:
                raise ChunkNotFoundError(f"Chunk with ID {chunk.id} not found")

            # Update secondary indexes if relationships changed
            if existing.document_id != chunk.document_id:
                self._document_index[existing.document_id].discard(chunk.id)
                self._document_index[chunk.document_id].add(chunk.id)

            if existing.library_id != chunk.library_id:
                self._library_index[existing.library_id].discard(chunk.id)
                self._library_index[chunk.library_id].add(chunk.id)

            # Store updated chunk
            self._chunks[chunk.id] = chunk

            return chunk

    def delete(self, chunk_id: UUID) -> bool:
        """Delete a chunk by its ID."""
        with self._lock.write_lock():
            chunk = self._chunks.get(chunk_id)
            if chunk is None:
                return False

            # Remove from primary storage
            del self._chunks[chunk_id]

            # Update secondary indexes
            self._document_index[chunk.document_id].discard(chunk_id)
            self._library_index[chunk.library_id].discard(chunk_id)

            return True

    def delete_by_document(self, document_id: UUID) -> int:
        """Delete all chunks in a document."""
        with self._lock.write_lock():
            chunk_ids = list(self._document_index.get(document_id, set()))

            # Delete each chunk (avoiding reentrancy issues)
            deleted_count = 0
            for chunk_id in chunk_ids:
                chunk = self._chunks.get(chunk_id)
                if chunk is not None:
                    # Remove from primary storage
                    del self._chunks[chunk_id]

                    # Update library index
                    self._library_index[chunk.library_id].discard(chunk_id)

                    deleted_count += 1

            # Clear the document index
            if document_id in self._document_index:
                self._document_index[document_id].clear()

            return deleted_count

    def delete_by_library(self, library_id: UUID) -> int:
        """Delete all chunks in a library."""
        with self._lock.write_lock():
            chunk_ids = list(self._library_index.get(library_id, set()))

            # Delete each chunk (avoiding reentrancy issues)
            deleted_count = 0
            for chunk_id in chunk_ids:
                chunk = self._chunks.get(chunk_id)
                if chunk is not None:
                    # Remove from primary storage
                    del self._chunks[chunk_id]

                    # Update document index
                    self._document_index[chunk.document_id].discard(chunk_id)

                    deleted_count += 1

            # Clear the library index
            if library_id in self._library_index:
                self._library_index[library_id].clear()

            return deleted_count

    def exists(self, chunk_id: UUID) -> bool:
        """Check if a chunk exists."""
        with self._lock.read_lock():
            return chunk_id in self._chunks

    def clear(self) -> None:
        """Clear all chunks (for testing purposes)."""
        with self._lock.write_lock():
            self._chunks.clear()
            self._document_index.clear()
            self._library_index.clear()
