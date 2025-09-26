"""Library schemas for API requests and responses."""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.config import settings
from app.domain import LibraryMetadata


class LibraryMetadataSchema(BaseModel):
    """Pydantic schema for LibraryMetadata."""

    model_config = ConfigDict(strict=True, extra="forbid", exclude_none=True)

    author: str | None = Field(None, max_length=settings.max_name_length, description="Library author")
    version: str | None = Field(None, max_length=50, description="Library version")
    tags: list[str] | None = Field(None, description="Library tags")
    created_by: str | None = Field(None, max_length=settings.max_name_length, description="Creator name")
    project: str | None = Field(None, max_length=settings.max_name_length, description="Project name")
    category: str | None = Field(None, max_length=100, description="Library category")
    is_public: bool | None = Field(None, description="Whether library is public")
    # Test/workflow fields
    test: bool | None = Field(None, description="Test flag")
    updated: bool | None = Field(None, description="Updated flag")
    original: bool | None = Field(None, description="Original flag")
    workflow: str | None = Field(None, max_length=100, description="Workflow type")

    def to_domain(self) -> LibraryMetadata:
        """Convert to domain LibraryMetadata."""
        return LibraryMetadata(
            author=self.author,
            version=self.version,
            tags=self.tags or [],  # Default to empty list if None
            created_by=self.created_by,
            project=self.project,
            category=self.category,
            is_public=self.is_public if self.is_public is not None else True,
            test=self.test,
            updated=self.updated,
            original=self.original,
            workflow=self.workflow,
        )

    @classmethod
    def from_domain(cls, metadata: LibraryMetadata) -> dict:
        """Create dict from domain LibraryMetadata, excluding default values."""
        return {
            k: v for k, v in {
                "author": metadata.author,
                "version": metadata.version,
                "tags": metadata.tags if metadata.tags else None,
                "created_by": metadata.created_by,
                "project": metadata.project,
                "category": metadata.category,
                "is_public": metadata.is_public if metadata.is_public is not True else None,
                "test": metadata.test,
                "updated": metadata.updated,
                "original": metadata.original,
                "workflow": metadata.workflow,
            }.items() if v is not None
        }


class LibraryBase(BaseModel):
    """Base schema for library data."""

    model_config = ConfigDict(strict=False, extra="forbid")

    name: str = Field(..., min_length=1, max_length=settings.max_name_length, description="Library name")
    description: str = Field(
        default="", max_length=1000, description="Library description"
    )
    metadata: LibraryMetadataSchema = Field(
        default_factory=LibraryMetadataSchema, description="Structured metadata"
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Library name cannot be empty or whitespace only")
        return v.strip()


class LibraryCreate(LibraryBase):
    """Schema for creating a new library."""

    # All fields inherited from LibraryBase
    pass


class LibraryUpdate(BaseModel):
    """Schema for updating an existing library."""

    model_config = ConfigDict(strict=False, extra="forbid")

    name: str | None = Field(
        None, min_length=1, max_length=settings.max_name_length, description="Library name"
    )
    description: str | None = Field(
        None, max_length=1000, description="Library description"
    )
    metadata: LibraryMetadataSchema | None = Field(
        None, description="Structured metadata"
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            raise ValueError("Library name cannot be empty or whitespace only")
        return v.strip() if v is not None else None


class LibraryOut(LibraryBase):
    """Schema for library responses."""

    id: UUID = Field(..., description="Unique library identifier")

    @classmethod
    def from_domain(cls, library) -> "LibraryOut":
        return cls(
            id=library.id,
            name=library.name,
            description=library.description,
            metadata=LibraryMetadataSchema.from_domain(library.metadata),
        )


class LibraryList(BaseModel):
    """Schema for paginated library listings."""

    model_config = ConfigDict(strict=False, extra="forbid")

    libraries: list[LibraryOut] = Field(..., description="List of libraries")
    total: int = Field(..., ge=0, description="Total number of libraries")
    limit: int | None = Field(None, ge=1, description="Number of items requested")
    offset: int = Field(0, ge=0, description="Number of items skipped")

    @classmethod
    def from_domain_list(
        cls, libraries, total: int, limit: int | None = None, offset: int = 0
    ) -> "LibraryList":
        return cls(
            libraries=[LibraryOut.from_domain(lib) for lib in libraries],
            total=total,
            limit=limit,
            offset=offset,
        )
