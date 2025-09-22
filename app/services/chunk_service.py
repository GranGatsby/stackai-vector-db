"""Chunk service for orchestrating chunk-related business operations.

This module contains the ChunkService class that implements the use cases
for chunk management, providing a clean interface between the API layer
and the domain/repository layers.
"""

from uuid import UUID

from app.domain import Chunk, ChunkMetadata, ChunkNotFoundError
from app.repositories.ports import (
    ChunkRepository,
    DocumentRepository,
    LibraryRepository,
)


class ChunkService:
    """Service class for chunk-related business operations.

    This service orchestrates chunk use cases, enforcing business rules
    and coordinating between the domain entities and repository layer.
    It provides cross-entity validations and handles embedding computation.

    Attributes:
        _chunk_repository: The chunk repository implementation
        _document_repository: The document repository for cross-validation
        _library_repository: The library repository for cross-validation
    """

    def __init__(
        self,
        chunk_repository: ChunkRepository,
        document_repository: DocumentRepository,
        library_repository: LibraryRepository,
    ) -> None:
        """Initialize the chunk service.

        Args:
            chunk_repository: The chunk repository implementation
            document_repository: The document repository for validation
            library_repository: The library repository for validation
        """
        self._chunk_repository = chunk_repository
        self._document_repository = document_repository
        self._library_repository = library_repository

    def list_chunks_by_document(
        self, document_id: UUID, limit: int = None, offset: int = 0
    ) -> tuple[list[Chunk], int]:
        """Retrieve chunks in a document with pagination.

        Args:
            document_id: The document ID to filter by
            limit: Maximum number of chunks to return (None for all)
            offset: Number of chunks to skip

        Returns:
            Tuple of (chunks list, total count)

        Raises:
            ValueError: If document_id doesn't exist
        """
        # Validate document exists
        if not self._document_repository.exists(document_id):
            raise ValueError(f"Document with ID {document_id} does not exist")

        chunks = self._chunk_repository.list_by_document(
            document_id, limit=limit, offset=offset
        )
        total = self._chunk_repository.count_by_document(document_id)
        return chunks, total

    def list_chunks_by_library(
        self, library_id: UUID, limit: int = None, offset: int = 0
    ) -> tuple[list[Chunk], int]:
        """Retrieve chunks in a library with pagination.

        Args:
            library_id: The library ID to filter by
            limit: Maximum number of chunks to return (None for all)
            offset: Number of chunks to skip

        Returns:
            Tuple of (chunks list, total count)

        Raises:
            ValueError: If library_id doesn't exist
        """
        # Validate library exists
        if not self._library_repository.exists(library_id):
            raise ValueError(f"Library with ID {library_id} does not exist")

        chunks = self._chunk_repository.list_by_library(
            library_id, limit=limit, offset=offset
        )
        total = self._chunk_repository.count_by_library(library_id)
        return chunks, total

    def get_chunk(self, chunk_id: UUID) -> Chunk:
        """Retrieve a chunk by its ID.

        Args:
            chunk_id: The unique identifier of the chunk

        Returns:
            The Chunk entity

        Raises:
            ChunkNotFoundError: If the chunk doesn't exist
        """
        chunk = self._chunk_repository.get_by_id(chunk_id)
        if chunk is None:
            raise ChunkNotFoundError(f"Chunk with ID {chunk_id} not found")
        return chunk

    def create_chunk(
        self,
        document_id: UUID,
        library_id: UUID,
        text: str,
        embedding: list[float] = None,
        start_index: int = 0,
        end_index: int = 0,
        metadata: ChunkMetadata | None = None,
        compute_embedding: bool = False,
    ) -> Chunk:
        """Create a new chunk.

        Args:
            document_id: The document ID where the chunk belongs
            library_id: The library ID where the chunk belongs
            text: The chunk text content
            embedding: Pre-computed embedding vector (optional)
            start_index: Starting character index in the source document
            end_index: Ending character index in the source document
            metadata: Additional chunk metadata
            compute_embedding: Whether to compute embedding automatically (placeholder)

        Returns:
            The created Chunk entity

        Raises:
            ValueError: If document_id or library_id don't exist, or text is invalid
        """
        # Validate document exists
        if not self._document_repository.exists(document_id):
            raise ValueError(f"Document with ID {document_id} does not exist")

        # Validate library exists
        if not self._library_repository.exists(library_id):
            raise ValueError(f"Library with ID {library_id} does not exist")

        # Cross-validate that document belongs to library
        document = self._document_repository.get_by_id(document_id)
        if document and document.library_id != library_id:
            raise ValueError(
                f"Document {document_id} does not belong to library {library_id}"
            )

        # Compute embedding if requested (placeholder implementation)
        final_embedding = embedding
        if compute_embedding and not embedding:
            final_embedding = self._compute_embedding(text)

        # Create the chunk using domain factory
        chunk = Chunk.create(
            document_id=document_id,
            library_id=library_id,
            text=text,
            embedding=final_embedding,
            start_index=start_index,
            end_index=end_index or (start_index + len(text.strip())),
            metadata=metadata,
        )

        # Store the chunk
        return self._chunk_repository.create(chunk)

    def update_chunk(
        self,
        chunk_id: UUID,
        text: str = None,
        embedding: list[float] = None,
        start_index: int = None,
        end_index: int = None,
        metadata: ChunkMetadata | None = None,
        compute_embedding: bool = False,
    ) -> Chunk:
        """Update an existing chunk with partial data.

        Args:
            chunk_id: The unique identifier of the chunk to update
            text: New text content (if provided)
            embedding: New embedding vector (if provided)
            start_index: New start index (if provided)
            end_index: New end index (if provided)
            metadata: New metadata (if provided)
            compute_embedding: Whether to recompute embedding if text changed

        Returns:
            The updated Chunk entity

        Raises:
            ChunkNotFoundError: If the chunk doesn't exist
        """
        # Get the existing chunk
        existing_chunk = self.get_chunk(chunk_id)

        # Compute new embedding if requested and text changed
        final_embedding = embedding
        if compute_embedding and text is not None and text != existing_chunk.text:
            final_embedding = self._compute_embedding(text)
        elif embedding is None:
            final_embedding = existing_chunk.embedding

        # Create updated chunk with new values
        updated_chunk = existing_chunk.update(
            text=text,
            embedding=final_embedding,
            start_index=start_index,
            end_index=end_index,
            metadata=metadata,
        )

        # Store the updated chunk
        return self._chunk_repository.update(updated_chunk)

    def delete_chunk(self, chunk_id: UUID) -> bool:
        """Delete a chunk by its ID.

        Args:
            chunk_id: The unique identifier of the chunk to delete

        Returns:
            True if the chunk was deleted, False if it didn't exist
        """
        return self._chunk_repository.delete(chunk_id)

    def delete_chunks_by_document(self, document_id: UUID) -> int:
        """Delete all chunks in a document (cascade operation).

        Args:
            document_id: The document ID to delete chunks from

        Returns:
            Number of chunks deleted
        """
        return self._chunk_repository.delete_by_document(document_id)

    def delete_chunks_by_library(self, library_id: UUID) -> int:
        """Delete all chunks in a library (cascade operation).

        Args:
            library_id: The library ID to delete chunks from

        Returns:
            Number of chunks deleted
        """
        return self._chunk_repository.delete_by_library(library_id)

    def chunk_exists(self, chunk_id: UUID) -> bool:
        """Check if a chunk exists.

        Args:
            chunk_id: The unique identifier to check

        Returns:
            True if the chunk exists, False otherwise
        """
        return self._chunk_repository.exists(chunk_id)

    def count_chunks_by_document(self, document_id: UUID) -> int:
        """Get the count of chunks in a document.

        Args:
            document_id: The document ID to count chunks for

        Returns:
            Number of chunks in the document

        Raises:
            ValueError: If document_id doesn't exist
        """
        # Validate document exists
        if not self._document_repository.exists(document_id):
            raise ValueError(f"Document with ID {document_id} does not exist")

        return self._chunk_repository.count_by_document(document_id)

    def count_chunks_by_library(self, library_id: UUID) -> int:
        """Get the count of chunks in a library.

        Args:
            library_id: The library ID to count chunks for

        Returns:
            Number of chunks in the library

        Raises:
            ValueError: If library_id doesn't exist
        """
        # Validate library exists
        if not self._library_repository.exists(library_id):
            raise ValueError(f"Library with ID {library_id} does not exist")

        return self._chunk_repository.count_by_library(library_id)

    def _compute_embedding(self, text: str) -> list[float]:
        """Compute embedding for the given text (placeholder implementation).

        This is a placeholder method that will be replaced with actual
        embedding computation using external services (e.g., Cohere API)
        in the embedding implementation phase.

        Args:
            text: The text to compute embedding for

        Returns:
            A placeholder embedding vector

        Note:
            This currently returns a dummy embedding. In the real implementation,
            this would call an embedding service and return the actual vector.
        """
        # Placeholder: return a dummy embedding based on text length
        # This will be replaced with actual embedding computation
        return [0.1] * min(768, max(1, len(text) // 10))  # Dummy 768-dim vector
