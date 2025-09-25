"""Chunk schemas for API requests and responses."""

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.domain import ChunkMetadata


class ChunkMetadataSchema(BaseModel):
    """Pydantic schema for ChunkMetadata."""

    model_config = ConfigDict(strict=True, extra="forbid", exclude_none=True)

    chunk_type: str | None = Field(
        None, max_length=100, description="Chunk type (paragraph, heading, table, etc.)"
    )
    section: str | None = Field(None, max_length=255, description="Section name")
    page_number: int | None = Field(None, ge=1, description="Page number")
    confidence: float | None = Field(
        None, ge=0.0, le=1.0, description="Extraction confidence"
    )
    language: str | None = Field(None, max_length=50, description="Chunk language")
    tags: list[str] | None = Field(None, description="Chunk tags")
    # Embedding fields
    embedding_model: str | None = Field(
        None, max_length=100, description="Embedding model used"
    )
    embedding_dim: int | None = Field(None, ge=1, description="Embedding dimension")
    similarity_threshold: float | None = Field(
        None, ge=0.0, description="Maximum distance threshold for search results (lower = more similar)"
    )
    # Processing fields
    processed_at: str | None = Field(
        None, description="Processing datetime (ISO string)"
    )
    is_indexed: bool | None = Field(None, description="Whether chunk is indexed")

    @field_validator("similarity_threshold")
    @classmethod
    def validate_similarity_threshold(cls, v: float | None) -> float | None:
        """Validate similarity threshold is reasonable for distance-based search."""
        if v is not None:
            if v < 0:
                raise ValueError("similarity_threshold must be non-negative")
            if v > 10:  # Very generous upper bound for distance-based similarity
                raise ValueError("similarity_threshold must be <= 10 (distances are typically 0-2)")
        return v

    def to_domain(self) -> ChunkMetadata:
        """Convert to domain ChunkMetadata."""
        return ChunkMetadata(
            chunk_type=self.chunk_type,
            section=self.section,
            page_number=self.page_number,
            confidence=self.confidence,
            language=self.language,
            tags=self.tags or [],
            embedding_model=self.embedding_model,
            embedding_dim=self.embedding_dim,
            similarity_threshold=self.similarity_threshold,
            processed_at=self.processed_at,
            is_indexed=self.is_indexed if self.is_indexed is not None else False,
        )

    @classmethod
    def dict_to_domain(cls, metadata_dict: dict) -> ChunkMetadata:
        """Convert a dict to domain ChunkMetadata."""
        return ChunkMetadata(
            chunk_type=metadata_dict.get("chunk_type"),
            section=metadata_dict.get("section"),
            page_number=metadata_dict.get("page_number"),
            confidence=metadata_dict.get("confidence"),
            language=metadata_dict.get("language"),
            tags=metadata_dict.get("tags", []),
            embedding_model=metadata_dict.get("embedding_model"),
            embedding_dim=metadata_dict.get("embedding_dim"),
            similarity_threshold=metadata_dict.get("similarity_threshold"),
            processed_at=metadata_dict.get("processed_at"),
            is_indexed=metadata_dict.get("is_indexed", False),
        )

    @classmethod
    def from_domain(cls, metadata: ChunkMetadata) -> dict:
        """Create a dict from domain ChunkMetadata for API responses.

        This returns only fields that have non-default values to maintain
        backward compatibility with existing API tests.
        """
        data = {}
        if metadata.chunk_type is not None:
            data["chunk_type"] = metadata.chunk_type
        if metadata.section is not None:
            data["section"] = metadata.section
        if metadata.page_number is not None:
            data["page_number"] = metadata.page_number
        if metadata.confidence is not None:
            data["confidence"] = metadata.confidence
        if metadata.language is not None:
            data["language"] = metadata.language
        if metadata.tags:  # Only if not empty list
            data["tags"] = metadata.tags
        if metadata.embedding_model is not None:
            data["embedding_model"] = metadata.embedding_model
        if metadata.embedding_dim is not None:
            data["embedding_dim"] = metadata.embedding_dim
        if metadata.similarity_threshold is not None:
            data["similarity_threshold"] = metadata.similarity_threshold
        if metadata.processed_at is not None:
            data["processed_at"] = metadata.processed_at
        if metadata.is_indexed is not False:  # Only if not default
            data["is_indexed"] = metadata.is_indexed

        return data


class ChunkBase(BaseModel):
    """Base schema for chunk data."""

    model_config = ConfigDict(strict=False, extra="forbid")

    text: str = Field(..., min_length=1, description="Chunk text content")
    embedding: list[float] = Field(default_factory=list, description="Embedding vector")
    start_index: int = Field(
        0, ge=0, description="Starting character index in document"
    )
    end_index: int = Field(0, ge=0, description="Ending character index in document")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Validate chunk text."""
        if not v.strip():
            raise ValueError("Chunk text cannot be empty or whitespace only")
        return v.strip()

    @field_validator("end_index")
    @classmethod
    def validate_end_index(cls, v: int, info) -> int:
        """Validate end_index is >= start_index."""
        if hasattr(info, "data") and "start_index" in info.data:
            start_index = info.data["start_index"]
            if v < start_index:
                raise ValueError("end_index must be >= start_index")
        return v


class ChunkCreateInDocument(ChunkBase):
    """Schema for creating a new chunk in a document."""

    compute_embedding: bool = Field(
        False, description="Whether to compute embedding automatically"
    )

    # All other fields inherited from ChunkBase (text, embedding, start_index, end_index, metadata)


class ChunkUpdate(BaseModel):
    """Schema for updating an existing chunk.

    All fields are optional to support partial updates.
    """

    model_config = ConfigDict(strict=False, extra="forbid")

    text: str | None = Field(None, min_length=1, description="Chunk text content")
    embedding: list[float] | None = Field(None, description="Embedding vector")
    start_index: int | None = Field(None, ge=0, description="Starting character index")
    end_index: int | None = Field(None, ge=0, description="Ending character index")
    metadata: dict[str, Any] | None = Field(None, description="Chunk metadata")
    compute_embedding: bool = Field(
        False, description="Whether to recompute embedding if text changed"
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str | None) -> str | None:
        """Validate chunk text if provided."""
        if v is not None and not v.strip():
            raise ValueError("Chunk text cannot be empty or whitespace only")
        return v.strip() if v is not None else None

    @field_validator("end_index")
    @classmethod
    def validate_end_index(cls, v: int | None, info) -> int | None:
        """Validate end_index is >= start_index if both are provided."""
        if v is not None and hasattr(info, "data") and "start_index" in info.data:
            start_index = info.data.get("start_index")
            if start_index is not None and v < start_index:
                raise ValueError("end_index must be >= start_index")
        return v


class ChunkRead(ChunkBase):
    """Schema for chunk responses."""

    id: UUID = Field(..., description="Unique chunk identifier")
    document_id: UUID = Field(..., description="Document ID that contains this chunk")
    library_id: UUID = Field(..., description="Library ID that contains this chunk")

    @classmethod
    def from_domain(cls, chunk) -> "ChunkRead":
        """Create a ChunkRead from a domain Chunk entity.

        Args:
            chunk: The domain Chunk entity

        Returns:
            A ChunkRead schema instance
        """
        return cls(
            id=chunk.id,
            document_id=chunk.document_id,
            library_id=chunk.library_id,
            text=chunk.text,
            embedding=chunk.embedding,
            start_index=chunk.start_index,
            end_index=chunk.end_index,
            metadata=ChunkMetadataSchema.from_domain(chunk.metadata),
        )


class ChunkList(BaseModel):
    """Schema for paginated chunk listings."""

    model_config = ConfigDict(strict=False, extra="forbid")

    chunks: list[ChunkRead] = Field(..., description="List of chunks")
    total: int = Field(..., ge=0, description="Total number of chunks")
    limit: int | None = Field(None, ge=1, description="Number of items requested")
    offset: int = Field(0, ge=0, description="Number of items skipped")

    @classmethod
    def from_domain_list(
        cls, chunks, total: int, limit: int | None = None, offset: int = 0
    ) -> "ChunkList":
        """Create a ChunkList from a list of domain Chunk entities.

        Args:
            chunks: List of domain Chunk entities
            total: Total count of chunks (before pagination)
            limit: Number of items requested
            offset: Number of items skipped

        Returns:
            A ChunkList schema instance
        """
        return cls(
            chunks=[ChunkRead.from_domain(chunk) for chunk in chunks],
            total=total,
            limit=limit,
            offset=offset,
        )
