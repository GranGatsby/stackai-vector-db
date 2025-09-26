"""Chunk service for orchestrating chunk-related business operations.

This module contains the ChunkService class that implements the use cases
for chunk management, providing a clean interface between the API layer
and the domain/repository layers.
"""

# Import for type hints only - will be injected as dependency
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from app.clients import EmbeddingClient, EmbeddingError, EmbeddingResult
from app.domain import Chunk, ChunkMetadata, ChunkNotFoundError
from app.repositories.ports import (
    ChunkRepository,
    DocumentRepository,
    LibraryRepository,
)

if TYPE_CHECKING:
    from app.services.index_service import IndexService


class ChunkService:
    """Service class for chunk-related business operations.

    This service orchestrates chunk use cases, enforcing business rules
    and coordinating between the domain entities and repository layer.
    It provides cross-entity validations and handles embedding computation.

    Attributes:
        _chunk_repository: The chunk repository implementation
        _document_repository: The document repository for cross-validation
        _library_repository: The library repository for cross-validation
        _embedding_client: The embedding client for computing embeddings
    """

    def __init__(
        self,
        chunk_repository: ChunkRepository,
        document_repository: DocumentRepository,
        library_repository: LibraryRepository,
        embedding_client: EmbeddingClient,
        index_service: "IndexService" = None,
    ) -> None:
        """Initialize the chunk service.

        Args:
            chunk_repository: The chunk repository implementation
            document_repository: The document repository for validation
            library_repository: The library repository for validation
            embedding_client: The embedding client for computing embeddings
            index_service: The index service for marking dirty (optional)
        """
        self._chunk_repository = chunk_repository
        self._document_repository = document_repository
        self._library_repository = library_repository
        self._embedding_client = embedding_client
        self._index_service = index_service

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

    def create_chunks(
        self,
        document_id: UUID,
        chunks_data: list[dict],
        compute_embedding: bool = False,
    ) -> list[Chunk]:
        """Create multiple chunks in a document.

        Args:
            document_id: The document ID where the chunks belong
            chunks_data: List of chunk data dictionaries containing text, embedding, etc.
            compute_embedding: Whether to compute embedding automatically for all chunks

        Returns:
            List of created Chunk entities

        Raises:
            ValueError: If document_id doesn't exist or chunk data is invalid
        """
        # Validate and get document
        document = self._document_repository.get_by_id(document_id)
        if document is None:
            raise ValueError(f"Document with ID {document_id} does not exist")

        library_id = document.library_id
        created_chunks = []

        # Compute embeddings in batch if requested
        embedding_result = None
        if compute_embedding:
            texts_to_embed = [
                chunk_data.get("text", "")
                for chunk_data in chunks_data
                if not chunk_data.get(
                    "embedding"
                )
            ]
            if texts_to_embed:
                try:
                    embedding_result = self._embedding_client.embed_texts(
                        texts_to_embed
                    )
                except EmbeddingError as e:
                    raise ValueError(f"Failed to compute embeddings: {e}") from e

        current_time = datetime.now().isoformat()

        # Process each chunk
        embedding_index = 0
        for chunk_data in chunks_data:
            text = chunk_data.get("text", "")
            embedding = chunk_data.get("embedding")
            start_index = chunk_data.get("start_index", 0)
            end_index = chunk_data.get("end_index", 0)
            metadata = chunk_data.get("metadata")

            # Use computed embedding if available
            final_embedding = embedding
            final_metadata = metadata

            if final_metadata is None:
                final_metadata = ChunkMetadata()

            if compute_embedding and not embedding and embedding_result:
                final_embedding = embedding_result.embeddings[embedding_index]
                embedding_index += 1

                # Create updated metadata with embedding info
                final_metadata = ChunkMetadata(
                    # Preserve existing metadata
                    chunk_type=final_metadata.chunk_type,
                    section=final_metadata.section,
                    page_number=final_metadata.page_number,
                    confidence=final_metadata.confidence,
                    language=final_metadata.language,
                    tags=final_metadata.tags,
                    similarity_threshold=final_metadata.similarity_threshold,
                    processed_at=current_time,
                    embedding_model=embedding_result.model_name,
                    embedding_dim=embedding_result.embedding_dim,
                )
            else:
                # Update metadata with current processing time (even if no embedding computed)
                final_metadata = ChunkMetadata(
                    # Preserve existing metadata
                    chunk_type=final_metadata.chunk_type,
                    section=final_metadata.section,
                    page_number=final_metadata.page_number,
                    confidence=final_metadata.confidence,
                    language=final_metadata.language,
                    tags=final_metadata.tags,
                    similarity_threshold=final_metadata.similarity_threshold,
                    processed_at=current_time,
                    embedding_model=final_metadata.embedding_model,
                    embedding_dim=final_metadata.embedding_dim,
                )

            # Create the chunk using domain factory
            chunk = Chunk.create(
                document_id=document_id,
                library_id=library_id,
                text=text,
                embedding=final_embedding,
                start_index=start_index,
                end_index=end_index or (start_index + len(text.strip())),
                metadata=final_metadata,
            )

            # Store the chunk
            created_chunk = self._chunk_repository.create(chunk)
            created_chunks.append(created_chunk)

        # Mark index as dirty after chunk creation (once for all chunks)
        if self._index_service and created_chunks:
            try:
                self._index_service.mark_dirty(library_id)
            except Exception:
                # Don't fail chunk creation if index marking fails
                pass

        return created_chunks

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

        current_time = datetime.now().isoformat()

        # Compute new embedding if requested and text changed
        final_embedding = embedding
        final_metadata = metadata if metadata is not None else existing_chunk.metadata

        # Handle embedding computation if requested
        embedding_model = None
        embedding_dim = None

        if compute_embedding and text is not None and text != existing_chunk.text:
            embedding_result = self._compute_embedding(text)
            final_embedding = embedding_result.single_embedding
            embedding_model = embedding_result.model_name
            embedding_dim = embedding_result.embedding_dim
        elif embedding is None:
            final_embedding = existing_chunk.embedding

        # Always ensure metadata exists and update it with current processing time
        if final_metadata is None:
            final_metadata = ChunkMetadata()

        # Create updated metadata with current processing time
        final_metadata = ChunkMetadata(
            # Preserve existing metadata
            chunk_type=final_metadata.chunk_type,
            section=final_metadata.section,
            page_number=final_metadata.page_number,
            confidence=final_metadata.confidence,
            language=final_metadata.language,
            tags=final_metadata.tags,
            similarity_threshold=final_metadata.similarity_threshold,
            processed_at=current_time,
            embedding_model=embedding_model or final_metadata.embedding_model,
            embedding_dim=embedding_dim or final_metadata.embedding_dim,
        )

        # Create updated chunk with new values
        updated_chunk = existing_chunk.update(
            text=text,
            embedding=final_embedding,
            start_index=start_index,
            end_index=end_index,
            metadata=final_metadata,
        )

        # Store the updated chunk
        updated_result = self._chunk_repository.update(updated_chunk)

        # Mark index as dirty after chunk update
        if self._index_service:
            try:
                self._index_service.mark_dirty(updated_result.library_id)
            except Exception:
                # Don't fail chunk update if index marking fails
                pass

        return updated_result

    def delete_chunk(self, chunk_id: UUID) -> bool:
        """Delete a chunk by its ID.

        Args:
            chunk_id: The unique identifier of the chunk to delete

        Returns:
            True if the chunk was deleted, False if it didn't exist
        """
        # Get chunk before deletion for library_id
        chunk = self._chunk_repository.get_by_id(chunk_id)
        deleted = self._chunk_repository.delete(chunk_id)

        # Mark index as dirty after chunk deletion
        if deleted and chunk and self._index_service:
            try:
                self._index_service.mark_dirty(chunk.library_id)
            except Exception:
                # Don't fail chunk deletion if index marking fails
                pass

        return deleted

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

    def _compute_embedding(self, text: str) -> EmbeddingResult:
        """Compute embedding for the given text using the embedding client.

        Args:
            text: The text to compute embedding for

        Returns:
            EmbeddingResult containing the embedding vector and metadata

        Raises:
            ValueError: If embedding computation fails
        """
        try:
            return self._embedding_client.embed_text(text)
        except EmbeddingError as e:
            raise ValueError(f"Failed to compute embedding: {e}") from e
