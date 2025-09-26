"""Chunk service for business operations."""

from __future__ import annotations

# Import for type hints only - will be injected as dependency
from contextlib import suppress
from datetime import datetime
from typing import TYPE_CHECKING

from app.clients import EmbeddingClient, EmbeddingError, EmbeddingResult
from app.domain import Chunk, ChunkMetadata, ChunkNotFoundError

if TYPE_CHECKING:
    from uuid import UUID

    from app.repositories.ports import (
        ChunkRepository,
        DocumentRepository,
        LibraryRepository,
    )
    from app.services.index_service import IndexService


class ChunkService:
    """Service for chunk business operations with embedding computation."""

    def __init__(
        self,
        chunk_repository: ChunkRepository,
        document_repository: DocumentRepository,
        library_repository: LibraryRepository,
        embedding_client: EmbeddingClient,
        index_service: IndexService = None,
    ) -> None:
        self._chunk_repository = chunk_repository
        self._document_repository = document_repository
        self._library_repository = library_repository
        self._embedding_client = embedding_client
        self._index_service = index_service

    def list_chunks_by_document(
        self, document_id: UUID, limit: int | None = None, offset: int = 0
    ) -> tuple[list[Chunk], int]:
        if not self._document_repository.exists(document_id):
            raise ValueError(f"Document with ID {document_id} does not exist")

        chunks = self._chunk_repository.list_by_document(
            document_id, limit=limit, offset=offset
        )
        total = self._chunk_repository.count_by_document(document_id)
        return chunks, total

    def list_chunks_by_library(
        self, library_id: UUID, limit: int | None = None, offset: int = 0
    ) -> tuple[list[Chunk], int]:
        if not self._library_repository.exists(library_id):
            raise ValueError(f"Library with ID {library_id} does not exist")

        chunks = self._chunk_repository.list_by_library(
            library_id, limit=limit, offset=offset
        )
        total = self._chunk_repository.count_by_library(library_id)
        return chunks, total

    def get_chunk(self, chunk_id: UUID) -> Chunk:
        chunk = self._chunk_repository.get_by_id(chunk_id)
        if not chunk:
            raise ChunkNotFoundError(f"Chunk with ID {chunk_id} not found")
        return chunk

    def create_chunks(
        self,
        document_id: UUID,
        chunks_data: list[dict],
        compute_embedding: bool = False,
    ) -> list[Chunk]:
        document = self._document_repository.get_by_id(document_id)
        if not document:
            raise ValueError(f"Document with ID {document_id} does not exist")

        embedding_result = self._compute_batch_embeddings(chunks_data, compute_embedding)
        current_time = datetime.now().isoformat()
        
        created_chunks = []
        embedding_index = 0
        
        for chunk_data in chunks_data:
            final_embedding, final_metadata = self._process_chunk_data(
                chunk_data, embedding_result, embedding_index, current_time
            )
            
            if compute_embedding and not chunk_data.get("embedding") and embedding_result:
                embedding_index += 1

            chunk = Chunk.create(
                document_id=document_id,
                library_id=document.library_id,
                text=chunk_data.get("text", ""),
                embedding=final_embedding,
                start_index=chunk_data.get("start_index", 0),
                end_index=chunk_data.get("end_index") or (
                    chunk_data.get("start_index", 0) + len(chunk_data.get("text", "").strip())
                ),
                metadata=final_metadata,
            )

            created_chunks.append(self._chunk_repository.create(chunk))

        if self._index_service and created_chunks:
            with suppress(Exception):
                self._index_service.mark_dirty(document.library_id)

        return created_chunks

    def update_chunk(
        self,
        chunk_id: UUID,
        text: str | None = None,
        embedding: list[float] | None = None,
        start_index: int | None = None,
        end_index: int | None = None,
        metadata: ChunkMetadata | None = None,
        compute_embedding: bool = False,
    ) -> Chunk:
        existing_chunk = self.get_chunk(chunk_id)
        current_time = datetime.now().isoformat()

        final_embedding, embedding_model, embedding_dim = self._handle_embedding_update(
            existing_chunk, text, embedding, compute_embedding
        )
        
        final_metadata = self._update_chunk_metadata(
            metadata or existing_chunk.metadata,
            current_time,
            embedding_model,
            embedding_dim
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
            with suppress(Exception):
                self._index_service.mark_dirty(updated_result.library_id)

        return updated_result

    def delete_chunk(self, chunk_id: UUID) -> bool:
        chunk = self._chunk_repository.get_by_id(chunk_id)
        deleted = self._chunk_repository.delete(chunk_id)

        # Mark index as dirty after chunk deletion
        if deleted and chunk and self._index_service:
            with suppress(Exception):
                self._index_service.mark_dirty(chunk.library_id)

        return deleted

    def delete_chunks_by_document(self, document_id: UUID) -> int:
        return self._chunk_repository.delete_by_document(document_id)

    def delete_chunks_by_library(self, library_id: UUID) -> int:
        return self._chunk_repository.delete_by_library(library_id)

    def chunk_exists(self, chunk_id: UUID) -> bool:
        return self._chunk_repository.exists(chunk_id)

    def count_chunks_by_document(self, document_id: UUID) -> int:
        if not self._document_repository.exists(document_id):
            raise ValueError(f"Document with ID {document_id} does not exist")
        return self._chunk_repository.count_by_document(document_id)

    def count_chunks_by_library(self, library_id: UUID) -> int:
        if not self._library_repository.exists(library_id):
            raise ValueError(f"Library with ID {library_id} does not exist")
        return self._chunk_repository.count_by_library(library_id)

    def _compute_batch_embeddings(
        self, chunks_data: list[dict], compute_embedding: bool
    ) -> EmbeddingResult | None:
        if not compute_embedding:
            return None
            
        texts_to_embed = [
            chunk_data.get("text", "")
            for chunk_data in chunks_data
            if not chunk_data.get("embedding")
        ]
        
        if not texts_to_embed:
            return None
            
        try:
            return self._embedding_client.embed_texts(texts_to_embed)
        except EmbeddingError as e:
            raise ValueError(f"Failed to compute embeddings: {e}") from e
    
    def _process_chunk_data(
        self, 
        chunk_data: dict, 
        embedding_result: EmbeddingResult | None, 
        embedding_index: int,
        current_time: str
    ) -> tuple[list[float] | None, ChunkMetadata]:
        embedding = chunk_data.get("embedding")
        metadata = chunk_data.get("metadata") or ChunkMetadata()
        
        if embedding_result and not embedding:
            embedding = embedding_result.embeddings[embedding_index]
            metadata = ChunkMetadata(
                chunk_type=metadata.chunk_type,
                section=metadata.section,
                page_number=metadata.page_number,
                confidence=metadata.confidence,
                language=metadata.language,
                tags=metadata.tags,
                similarity_threshold=metadata.similarity_threshold,
                processed_at=current_time,
                embedding_model=embedding_result.model_name,
                embedding_dim=embedding_result.embedding_dim,
            )
        else:
            metadata = ChunkMetadata(
                chunk_type=metadata.chunk_type,
                section=metadata.section,
                page_number=metadata.page_number,
                confidence=metadata.confidence,
                language=metadata.language,
                tags=metadata.tags,
                similarity_threshold=metadata.similarity_threshold,
                processed_at=current_time,
                embedding_model=metadata.embedding_model,
                embedding_dim=metadata.embedding_dim,
            )
        
        return embedding, metadata
    
    def _handle_embedding_update(
        self, existing_chunk: Chunk, text: str | None, embedding: list[float] | None, compute_embedding: bool
    ) -> tuple[list[float] | None, str | None, int | None]:
        if compute_embedding and text and text != existing_chunk.text:
            try:
                embedding_result = self._embedding_client.embed_text(text)
                return (
                    embedding_result.single_embedding,
                    embedding_result.model_name,
                    embedding_result.embedding_dim
                )
            except EmbeddingError as e:
                raise ValueError(f"Failed to compute embedding: {e}") from e
        
        return embedding or existing_chunk.embedding, None, None
    
    def _update_chunk_metadata(
        self, 
        base_metadata: ChunkMetadata, 
        current_time: str, 
        embedding_model: str | None, 
        embedding_dim: int | None
    ) -> ChunkMetadata:
        return ChunkMetadata(
            chunk_type=base_metadata.chunk_type,
            section=base_metadata.section,
            page_number=base_metadata.page_number,
            confidence=base_metadata.confidence,
            language=base_metadata.language,
            tags=base_metadata.tags,
            similarity_threshold=base_metadata.similarity_threshold,
            processed_at=current_time,
            embedding_model=embedding_model or base_metadata.embedding_model,
            embedding_dim=embedding_dim or base_metadata.embedding_dim,
        )
