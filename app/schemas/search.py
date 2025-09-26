"""Search schemas for API requests and responses."""

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.config import settings


class SearchByTextRequest(BaseModel):
    """Schema for text-based search requests."""

    model_config = ConfigDict(strict=True, extra="forbid")

    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Text query to search for similar chunks",
    )
    k: int = Field(
        10,
        ge=1,
        le=settings.max_knn_results,
        description="Number of nearest neighbors to return",
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Search text cannot be empty or whitespace only")
        return v.strip()


class SearchByVectorRequest(BaseModel):
    """Schema for embedding-based search requests."""

    model_config = ConfigDict(strict=True, extra="forbid")

    embedding: list[float] = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Query embedding vector for similarity search",
    )
    k: int = Field(
        10,
        ge=1,
        le=settings.max_knn_results,
        description="Number of nearest neighbors to return",
    )

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("Embedding vector cannot be empty")

        for i, val in enumerate(v):
            if not isinstance(val, (int, float)):
                raise ValueError(f"Embedding value at index {i} must be a number")
            if not (-1e6 <= val <= 1e6):
                raise ValueError(
                    f"Embedding value at index {i} is out of reasonable range"
                )
        return v


class SearchHit(BaseModel):
    """Schema for individual search result hits."""

    model_config = ConfigDict(strict=True, extra="forbid")

    chunk_id: UUID = Field(..., description="UUID of the matching chunk")
    score: float = Field(
        ..., ge=0.0, description="Similarity distance (lower = more similar)"
    )
    metadata: dict[str, Any] | None = Field(
        None, description="Optional chunk metadata for additional context"
    )


class SearchResult(BaseModel):
    """Schema for complete search results."""

    model_config = ConfigDict(strict=True, extra="forbid")

    hits: list[SearchHit] = Field(
        ..., description="List of matching chunks ordered by relevance (best first)"
    )
    total: int = Field(..., ge=0, description="Total number of results returned")
    library_id: UUID = Field(..., description="UUID of the searched library")
    algorithm: str = Field(..., description="Index algorithm used for search")
    index_size: int = Field(
        ..., ge=0, description="Total number of vectors in the searched index"
    )
    embedding_dim: int = Field(
        ..., ge=1, le=10000, description="Dimension of embeddings used in search"
    )
    query_embedding: list[float] | None = Field(
        None, description="Query embedding used (optional, may be large)"
    )

    @field_validator("hits")
    @classmethod
    def validate_hits_sorted(cls, v: list[SearchHit]) -> list[SearchHit]:
        if len(v) > 1:
            for i in range(1, len(v)):
                if v[i].score < v[i - 1].score:
                    raise ValueError("Search hits must be sorted by score (ascending)")
        return v

    @field_validator("total")
    @classmethod
    def validate_total_matches_hits(cls, v: int, info) -> int:
        if hasattr(info, "data") and "hits" in info.data:
            hits = info.data["hits"]
            if v != len(hits):
                raise ValueError("Total must equal the number of hits")
        return v

    @property
    def is_empty(self) -> bool:
        return self.total == 0

    @property
    def best_score(self) -> float | None:
        return self.hits[0].score if self.hits else None

    @property
    def worst_score(self) -> float | None:
        return self.hits[-1].score if self.hits else None
