"""Chunk schemas for API requests and responses."""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.config import settings
from app.domain import ChunkMetadata


class ChunkMetadataSchema(BaseModel):
    """Pydantic schema for ChunkMetadata."""

    model_config = ConfigDict(strict=True, extra="forbid", exclude_none=True)

    chunk_type: str | None = Field(
        None, max_length=100, description="Chunk type (paragraph, heading, table, etc.)"
    )
    section: str | None = Field(None, max_length=settings.max_name_length, description="Section name")
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
        None,
        ge=0.0,
        description="Maximum distance threshold for search results (lower = more similar)",
    )
    # Processing fields
    processed_at: str | None = Field(
        None, description="Processing datetime (ISO string)"
    )

    @field_validator("similarity_threshold")
    @classmethod
    def validate_similarity_threshold(cls, v: float | None) -> float | None:
        """Validate similarity threshold is reasonable for distance-based search."""
        if v is not None:
            if v < 0:
                raise ValueError("similarity_threshold must be non-negative")
            if v > 10:  # Very generous upper bound for distance-based similarity
                raise ValueError(
                    "similarity_threshold must be <= 10 (distances are typically 0-2)"
                )
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
        )

    @classmethod
    def from_domain(cls, metadata: ChunkMetadata) -> dict:
        """Create dict from domain ChunkMetadata, excluding None values."""
        return {
            k: v for k, v in {
                "chunk_type": metadata.chunk_type,
                "section": metadata.section,
                "page_number": metadata.page_number,
                "confidence": metadata.confidence,
                "language": metadata.language,
                "tags": metadata.tags if metadata.tags else None,
                "embedding_model": metadata.embedding_model,
                "embedding_dim": metadata.embedding_dim,
                "similarity_threshold": metadata.similarity_threshold,
                "processed_at": metadata.processed_at,
            }.items() if v is not None
        }


class ChunkBase(BaseModel):
    """Base schema for chunk data."""

    model_config = ConfigDict(strict=False, extra="forbid")

    text: str = Field(..., min_length=1, description="Chunk text content")
    embedding: list[float] = Field(default_factory=list, description="Embedding vector")
    start_index: int = Field(
        0, ge=0, description="Starting character index in document"
    )
    end_index: int = Field(0, ge=0, description="Ending character index in document")
    metadata: ChunkMetadataSchema = Field(
        default_factory=ChunkMetadataSchema, description="Chunk metadata"
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Chunk text cannot be empty or whitespace only")
        return v.strip()

    @field_validator("end_index")
    @classmethod
    def validate_end_index(cls, v: int, info) -> int:
        if hasattr(info, "data") and "start_index" in info.data:
            start_index = info.data["start_index"]
            if v < start_index:
                raise ValueError("end_index must be >= start_index")
        return v


class ChunkCreateInDocument(BaseModel):
    """Schema for creating one or more chunks in a document."""

    model_config = ConfigDict(strict=False, extra="forbid")

    chunks: list[ChunkBase] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of chunks to create (1-100 chunks)",
    )
    compute_embedding: bool = Field(
        False, description="Whether to compute embedding automatically for all chunks"
    )

    @field_validator("chunks")
    @classmethod
    def validate_chunks_count(cls, v: list[ChunkBase]) -> list[ChunkBase]:
        """Validate that we have between 1 and 100 chunks."""
        if not v:
            raise ValueError("At least one chunk must be provided")
        if len(v) > 100:
            raise ValueError("Cannot create more than 100 chunks at once")
        return v


class ChunkUpdate(BaseModel):
    """Schema for updating an existing chunk."""

    model_config = ConfigDict(strict=False, extra="forbid")

    text: str | None = Field(None, min_length=1, description="Chunk text content")
    embedding: list[float] | None = Field(None, description="Embedding vector")
    start_index: int | None = Field(None, ge=0, description="Starting character index")
    end_index: int | None = Field(None, ge=0, description="Ending character index")
    metadata: ChunkMetadataSchema | None = Field(None, description="Chunk metadata")
    compute_embedding: bool = Field(
        False, description="Whether to recompute embedding if text changed"
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str | None) -> str | None:
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
        return cls(
            chunks=[ChunkRead.from_domain(chunk) for chunk in chunks],
            total=total,
            limit=limit,
            offset=offset,
        )


class ChunkCreateResponse(BaseModel):
    """Schema for chunk creation response."""

    model_config = ConfigDict(strict=False, extra="forbid")

    chunks: list[ChunkRead] = Field(..., description="List of created chunks")
    total_created: int = Field(..., ge=1, description="Total number of chunks created")

    @classmethod
    def from_domain_list(cls, chunks) -> "ChunkCreateResponse":
        return cls(
            chunks=[ChunkRead.from_domain(chunk) for chunk in chunks],
            total_created=len(chunks),
        )
