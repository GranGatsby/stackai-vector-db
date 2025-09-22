"""Document schemas for API requests and responses."""

from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.domain import DocumentMetadata


class DocumentMetadataSchema(BaseModel):
    """Pydantic schema for DocumentMetadata."""
    
    model_config = ConfigDict(strict=True, extra="forbid", exclude_none=True)
    
    author: Optional[str] = Field(None, max_length=255, description="Document author")
    source: Optional[str] = Field(None, max_length=500, description="Document source")
    language: Optional[str] = Field(None, max_length=50, description="Document language")
    format: Optional[str] = Field(None, max_length=50, description="Document format (pdf, txt, markdown, etc.)")
    created_at: Optional[str] = Field(None, description="Creation datetime (ISO string)")
    modified_at: Optional[str] = Field(None, description="Modification datetime (ISO string)")
    tags: Optional[list[str]] = Field(None, description="Document tags")
    category: Optional[str] = Field(None, max_length=100, description="Document category")
    is_public: Optional[bool] = Field(None, description="Whether document is public")
    # Processing fields
    processed: Optional[bool] = Field(None, description="Processing flag")
    chunk_count: Optional[int] = Field(None, ge=0, description="Number of chunks")
    word_count: Optional[int] = Field(None, ge=0, description="Word count")

    def to_domain(self) -> DocumentMetadata:
        """Convert to domain DocumentMetadata."""
        return DocumentMetadata(
            author=self.author,
            source=self.source,
            language=self.language,
            format=self.format,
            created_at=self.created_at,
            modified_at=self.modified_at,
            tags=self.tags or [],
            category=self.category,
            is_public=self.is_public if self.is_public is not None else True,
            processed=self.processed,
            chunk_count=self.chunk_count,
            word_count=self.word_count,
        )

    @classmethod
    def dict_to_domain(cls, metadata_dict: dict) -> DocumentMetadata:
        """Convert a dict to domain DocumentMetadata."""
        return DocumentMetadata(
            author=metadata_dict.get("author"),
            source=metadata_dict.get("source"),
            language=metadata_dict.get("language"),
            format=metadata_dict.get("format"),
            created_at=metadata_dict.get("created_at"),
            modified_at=metadata_dict.get("modified_at"),
            tags=metadata_dict.get("tags", []),
            category=metadata_dict.get("category"),
            is_public=metadata_dict.get("is_public", True),
            processed=metadata_dict.get("processed"),
            chunk_count=metadata_dict.get("chunk_count"),
            word_count=metadata_dict.get("word_count"),
        )

    @classmethod
    def from_domain(cls, metadata: DocumentMetadata) -> dict:
        """Create a dict from domain DocumentMetadata for API responses.
        
        This returns only fields that have non-default values to maintain
        backward compatibility with existing API tests.
        """
        data = {}
        if metadata.author is not None:
            data["author"] = metadata.author
        if metadata.source is not None:
            data["source"] = metadata.source
        if metadata.language is not None:
            data["language"] = metadata.language
        if metadata.format is not None:
            data["format"] = metadata.format
        if metadata.created_at is not None:
            data["created_at"] = metadata.created_at
        if metadata.modified_at is not None:
            data["modified_at"] = metadata.modified_at
        if metadata.tags:  # Only if not empty list
            data["tags"] = metadata.tags
        if metadata.category is not None:
            data["category"] = metadata.category
        if metadata.is_public is not True:  # Only if not default
            data["is_public"] = metadata.is_public
        if metadata.processed is not None:
            data["processed"] = metadata.processed
        if metadata.chunk_count is not None:
            data["chunk_count"] = metadata.chunk_count
        if metadata.word_count is not None:
            data["word_count"] = metadata.word_count
        
        return data


class DocumentBase(BaseModel):
    """Base schema for document data."""

    model_config = ConfigDict(strict=True, extra="forbid")

    title: str = Field(..., min_length=1, max_length=255, description="Document title")
    content: str = Field(default="", description="Document content")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Document metadata"
    )

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Validate document title."""
        if not v.strip():
            raise ValueError("Document title cannot be empty or whitespace only")
        return v.strip()


class DocumentCreate(DocumentBase):
    """Schema for creating a new document."""

    library_id: UUID = Field(..., description="Library ID where document will be created")

    # All other fields inherited from DocumentBase


class DocumentUpdate(BaseModel):
    """Schema for updating an existing document.

    All fields are optional to support partial updates.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    title: str | None = Field(
        None, min_length=1, max_length=255, description="Document title"
    )
    content: str | None = Field(None, description="Document content")
    metadata: dict[str, Any] | None = Field(None, description="Document metadata")

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str | None) -> str | None:
        """Validate document title if provided."""
        if v is not None and not v.strip():
            raise ValueError("Document title cannot be empty or whitespace only")
        return v.strip() if v is not None else None


class DocumentRead(DocumentBase):
    """Schema for document responses."""

    id: UUID = Field(..., description="Unique document identifier")
    library_id: UUID = Field(..., description="Library ID that contains this document")

    @classmethod
    def from_domain(cls, document) -> "DocumentRead":
        """Create a DocumentRead from a domain Document entity.

        Args:
            document: The domain Document entity

        Returns:
            A DocumentRead schema instance
        """
        return cls(
            id=document.id,
            library_id=document.library_id,
            title=document.title,
            content=document.content,
            metadata=DocumentMetadataSchema.from_domain(document.metadata),
        )


class DocumentList(BaseModel):
    """Schema for paginated document listings."""

    model_config = ConfigDict(strict=True, extra="forbid")

    documents: list[DocumentRead] = Field(..., description="List of documents")
    total: int = Field(..., ge=0, description="Total number of documents")
    limit: int | None = Field(None, ge=1, description="Number of items requested")
    offset: int = Field(0, ge=0, description="Number of items skipped")

    @classmethod
    def from_domain_list(
        cls, documents, total: int, limit: int | None = None, offset: int = 0
    ) -> "DocumentList":
        """Create a DocumentList from a list of domain Document entities.

        Args:
            documents: List of domain Document entities
            total: Total count of documents (before pagination)
            limit: Number of items requested
            offset: Number of items skipped

        Returns:
            A DocumentList schema instance
        """
        return cls(
            documents=[DocumentRead.from_domain(doc) for doc in documents],
            total=total,
            limit=limit,
            offset=offset,
        )
