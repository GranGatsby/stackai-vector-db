"""Library schemas for API requests and responses."""

from typing import Dict, Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LibraryBase(BaseModel):
    """Base schema for library data."""

    model_config = ConfigDict(strict=True, extra="forbid")

    name: str = Field(..., min_length=1, max_length=255, description="Library name")
    description: str = Field(
        default="", max_length=1000, description="Library description"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate library name."""
        if not v.strip():
            raise ValueError("Library name cannot be empty or whitespace only")
        return v.strip()


class LibraryCreate(LibraryBase):
    """Schema for creating a new library."""

    # All fields inherited from LibraryBase
    pass


class LibraryUpdate(BaseModel):
    """Schema for updating an existing library.

    All fields are optional to support partial updates.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    name: Optional[str] = Field(
        None, min_length=1, max_length=255, description="Library name"
    )
    description: Optional[str] = Field(
        None, max_length=1000, description="Library description"
    )
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: Optional[str]) -> Optional[str]:
        """Validate library name if provided."""
        if v is not None and not v.strip():
            raise ValueError("Library name cannot be empty or whitespace only")
        return v.strip() if v is not None else None


class LibraryOut(LibraryBase):
    """Schema for library responses."""

    id: UUID = Field(..., description="Unique library identifier")

    @classmethod
    def from_domain(cls, library) -> "LibraryOut":
        """Create a LibraryOut from a domain Library entity.

        Args:
            library: The domain Library entity

        Returns:
            A LibraryOut schema instance
        """
        return cls(
            id=library.id,
            name=library.name,
            description=library.description,
            metadata=library.metadata,
        )


class LibraryList(BaseModel):
    """Schema for paginated library listings."""

    model_config = ConfigDict(strict=True, extra="forbid")

    libraries: list[LibraryOut] = Field(..., description="List of libraries")
    total: int = Field(..., ge=0, description="Total number of libraries")
    limit: Optional[int] = Field(None, ge=1, description="Number of items requested")
    offset: int = Field(0, ge=0, description="Number of items skipped")

    @classmethod
    def from_domain_list(
        cls, libraries, total: int, limit: Optional[int] = None, offset: int = 0
    ) -> "LibraryList":
        """Create a LibraryList from a list of domain Library entities.

        Args:
            libraries: List of domain Library entities
            total: Total count of libraries (before pagination)
            limit: Number of items requested
            offset: Number of items skipped

        Returns:
            A LibraryList schema instance
        """
        return cls(
            libraries=[LibraryOut.from_domain(lib) for lib in libraries],
            total=total,
            limit=limit,
            offset=offset,
        )
