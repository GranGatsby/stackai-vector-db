"""Index management schemas for API requests and responses."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Type alias for supported index algorithms
IndexAlgo = Literal["linear", "kdtree", "ivf"]


class IndexStatus(BaseModel):
    """Schema for index status information."""

    model_config = ConfigDict(strict=True, extra="forbid")

    library_id: UUID = Field(..., description="UUID of the library")
    algorithm: IndexAlgo = Field(..., description="Index algorithm being used")
    is_built: bool = Field(..., description="Whether the index has been built")
    is_dirty: bool = Field(..., description="Whether the index needs rebuilding")
    size: int = Field(..., ge=0, description="Number of vectors in the index")
    embedding_dim: int | None = Field(
        None,
        ge=1,
        le=10000,
        description="Dimension of embeddings (None if not determined)",
    )
    built_at: float | None = Field(
        None, description="Timestamp when index was last built (Unix timestamp)"
    )
    version: int = Field(..., ge=0, description="Version number of the current index")
    dirty_count: int = Field(
        ..., ge=0, description="Number of changes since last build"
    )

    @field_validator("version")
    @classmethod
    def validate_version_consistency(cls, v: int, info) -> int:
        if hasattr(info, "data") and "is_built" in info.data:
            is_built = info.data["is_built"]
            if is_built and v == 0:
                raise ValueError("Built indexes must have version >= 1")
            if not is_built and v > 0:
                raise ValueError("Unbuilt indexes should have version = 0")
        return v


class BuildIndexRequest(BaseModel):
    """Schema for index building requests."""

    model_config = ConfigDict(strict=True, extra="forbid")

    algorithm: IndexAlgo | None = Field(
        None, description="Index algorithm to use (uses current/default if None)"
    )


class BuildIndexResponse(BaseModel):
    """Schema for index building responses."""

    model_config = ConfigDict(strict=True, extra="forbid")

    library_id: UUID = Field(..., description="UUID of the library")
    algorithm: IndexAlgo = Field(..., description="Index algorithm used")
    size: int = Field(..., ge=0, description="Number of vectors indexed")
    embedding_dim: int = Field(
        ..., ge=1, le=10000, description="Dimension of embeddings"
    )
    built_at: float = Field(..., description="Timestamp when build completed")
    version: int = Field(..., ge=1, description="New version number of the index")
    build_duration: float | None = Field(
        None, ge=0.0, description="Build duration in seconds (if available)"
    )

    @classmethod
    def from_index_status(
        cls, status: IndexStatus, build_duration: float | None = None
    ) -> "BuildIndexResponse":
        if not status.is_built:
            raise ValueError("Cannot create BuildIndexResponse from unbuilt index")
        if status.embedding_dim is None:
            raise ValueError(
                "Cannot create BuildIndexResponse without embedding dimension"
            )
        if status.built_at is None:
            raise ValueError("Cannot create BuildIndexResponse without build timestamp")

        return cls(
            library_id=status.library_id,
            algorithm=status.algorithm,
            size=status.size,
            embedding_dim=status.embedding_dim,
            built_at=status.built_at,
            version=status.version,
            build_duration=build_duration,
        )
