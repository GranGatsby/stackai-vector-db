"""Library schemas for API requests and responses."""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.domain import LibraryMetadata


class LibraryMetadataSchema(BaseModel):
    """Pydantic schema for LibraryMetadata."""

    model_config = ConfigDict(strict=True, extra="forbid", exclude_none=True)

    author: str | None = Field(None, max_length=255, description="Library author")
    version: str | None = Field(None, max_length=50, description="Library version")
    tags: list[str] | None = Field(None, description="Library tags")
    created_by: str | None = Field(None, max_length=255, description="Creator name")
    project: str | None = Field(None, max_length=255, description="Project name")
    category: str | None = Field(
        None, max_length=100, description="Library category"
    )
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
            is_public=(
                self.is_public if self.is_public is not None else True
            ),  # Default to True
            test=self.test,
            updated=self.updated,
            original=self.original,
            workflow=self.workflow,
        )

    @classmethod
    def dict_to_domain(cls, metadata_dict: dict) -> LibraryMetadata:
        """Convert a dict to domain LibraryMetadata."""
        return LibraryMetadata(
            author=metadata_dict.get("author"),
            version=metadata_dict.get("version"),
            tags=metadata_dict.get("tags", []),
            created_by=metadata_dict.get("created_by"),
            project=metadata_dict.get("project"),
            category=metadata_dict.get("category"),
            is_public=metadata_dict.get("is_public", True),
            test=metadata_dict.get("test"),
            updated=metadata_dict.get("updated"),
            original=metadata_dict.get("original"),
            workflow=metadata_dict.get("workflow"),
        )

    @classmethod
    def from_domain(cls, metadata: LibraryMetadata) -> dict:
        """Create a dict from domain LibraryMetadata for API responses.

        This returns only fields that have non-default values to maintain
        backward compatibility with existing API tests.
        """
        data = {}
        if metadata.author is not None:
            data["author"] = metadata.author
        if metadata.version is not None:
            data["version"] = metadata.version
        if metadata.tags:  # Only if not empty list
            data["tags"] = metadata.tags
        if metadata.created_by is not None:
            data["created_by"] = metadata.created_by
        if metadata.project is not None:
            data["project"] = metadata.project
        if metadata.category is not None:
            data["category"] = metadata.category
        if metadata.is_public is not True:  # Only if not default
            data["is_public"] = metadata.is_public
        if metadata.test is not None:
            data["test"] = metadata.test
        if metadata.updated is not None:
            data["updated"] = metadata.updated
        if metadata.original is not None:
            data["original"] = metadata.original
        if metadata.workflow is not None:
            data["workflow"] = metadata.workflow

        return data


class LibraryBase(BaseModel):
    """Base schema for library data."""

    model_config = ConfigDict(strict=False, extra="forbid")

    name: str = Field(..., min_length=1, max_length=255, description="Library name")
    description: str = Field(
        default="", max_length=1000, description="Library description"
    )
    metadata: dict = Field(default_factory=dict, description="Structured metadata")

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

    model_config = ConfigDict(strict=False, extra="forbid")

    name: str | None = Field(
        None, min_length=1, max_length=255, description="Library name"
    )
    description: str | None = Field(
        None, max_length=1000, description="Library description"
    )
    metadata: dict | None = Field(None, description="Structured metadata")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str | None) -> str | None:
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
