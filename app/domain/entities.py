"""Domain entities for the vector database.

This module contains the core business entities that represent the fundamental
concepts of the vector database: Library, Document, and Chunk.
"""

import uuid
from dataclasses import dataclass, field
from uuid import UUID

from app.utils import (
    validate_index_range,
    validate_name_length,
    validate_non_empty_text,
    validate_non_negative,
    validate_title_length,
)


@dataclass(frozen=True)
class DocumentMetadata:
    """Fixed metadata schema for Document entities."""

    author: str | None = None
    source: str | None = None
    language: str | None = None
    format: str | None = None  # pdf, txt, markdown, etc.
    created_at: str | None = None  # ISO datetime string
    modified_at: str | None = None  # ISO datetime string
    tags: list[str] = field(default_factory=list)
    category: str | None = None
    is_public: bool = True
    # Processing fields
    processed: bool | None = None
    chunk_count: int | None = None
    word_count: int | None = None


@dataclass(frozen=True)
class ChunkMetadata:
    """Fixed metadata schema for Chunk entities."""

    chunk_type: str | None = None  # paragraph, heading, table, etc.
    section: str | None = None
    page_number: int | None = None
    confidence: float | None = None  # extraction confidence
    language: str | None = None
    tags: list[str] = field(default_factory=list)
    # Embedding fields
    embedding_model: str | None = None
    embedding_dim: int | None = None
    similarity_threshold: float | None = None
    # Processing fields
    processed_at: str | None = None  # ISO datetime string


@dataclass(frozen=True)
class LibraryMetadata:
    """Fixed metadata schema for Library entities."""

    author: str | None = None
    version: str | None = None
    tags: list[str] = field(default_factory=list)
    created_by: str | None = None
    project: str | None = None
    category: str | None = None
    is_public: bool = True
    # Test/workflow fields
    test: bool | None = None
    updated: bool | None = None
    original: bool | None = None
    workflow: str | None = None


@dataclass(frozen=True)
class Library:
    """Collection of documents with metadata and organizational structure.

    Attributes:
        id: Unique identifier for the library
        name: Human-readable name for the library
        description: Optional description of the library's purpose
        metadata: Structured metadata for the library
    """

    id: UUID
    name: str
    description: str = ""
    metadata: LibraryMetadata = field(default_factory=LibraryMetadata)

    def __post_init__(self) -> None:
        """Validate library invariants."""
        validate_name_length(self.name, "Library name")

    @classmethod
    def create(
        cls,
        name: str,
        description: str = "",
        metadata: LibraryMetadata | None = None,
    ) -> "Library":
        """Create new Library with generated ID.

        Raises:
            ValueError: If name is empty or too long
        """
        return cls(
            id=uuid.uuid4(),
            name=validate_name_length(name, "Library name"),
            description=description,
            metadata=metadata or LibraryMetadata(),
        )

    def update(
        self,
        name: str | None = None,
        description: str | None = None,
        metadata: LibraryMetadata | None = None,
    ) -> "Library":
        """Create new Library instance with updated fields.

        Raises:
            ValueError: If new name is empty or too long
        """
        updated_name = (
            validate_name_length(name, "Library name")
            if name is not None
            else self.name
        )
        return Library(
            id=self.id,
            name=updated_name,
            description=description if description is not None else self.description,
            metadata=metadata if metadata is not None else self.metadata,
        )


@dataclass(frozen=True)
class Document:
    """Document within a library, composed of multiple chunks.

    Attributes:
        id: Unique identifier for the document
        library_id: ID of the library containing this document
        title: Title or name of the document
        content: Full text content of the document
        metadata: Additional document metadata
    """

    id: UUID
    library_id: UUID
    title: str
    content: str = ""
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)

    def __post_init__(self) -> None:
        """Validate document invariants."""
        validate_title_length(self.title, "Document title")

    @classmethod
    def create(
        cls,
        library_id: UUID,
        title: str,
        content: str = "",
        metadata: DocumentMetadata | None = None,
    ) -> "Document":
        """Create new Document with generated ID."""
        return cls(
            id=uuid.uuid4(),
            library_id=library_id,
            title=validate_title_length(title, "Document title"),
            content=content,
            metadata=metadata or DocumentMetadata(),
        )

    def update(
        self,
        title: str | None = None,
        content: str | None = None,
        metadata: DocumentMetadata | None = None,
    ) -> "Document":
        """Create new Document instance with updated fields.

        Raises:
            ValueError: If new title is empty or too long
        """
        updated_title = (
            validate_title_length(title, "Document title")
            if title is not None
            else self.title
        )
        return Document(
            id=self.id,
            library_id=self.library_id,
            title=updated_title,
            content=content if content is not None else self.content,
            metadata=metadata if metadata is not None else self.metadata,
        )


@dataclass(frozen=True)
class Chunk:
    """Text chunk with embedding and metadata for vector search.

    Attributes:
        id: Unique identifier for the chunk
        document_id: ID of the parent document
        library_id: ID of the parent library
        text: The text content of this chunk
        embedding: Vector embedding of the text (list of floats)
        start_index: Starting character index in the source document
        end_index: Ending character index in the source document
        metadata: Additional chunk metadata
    """

    id: UUID
    document_id: UUID
    library_id: UUID
    text: str
    embedding: list[float] | None = None
    start_index: int = 0
    end_index: int = 0
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)

    def __post_init__(self) -> None:
        """Validate chunk invariants."""
        validate_non_empty_text(self.text, "Chunk text cannot be empty")
        validate_non_negative(self.start_index, "Chunk start_index")
        validate_index_range(self.start_index, self.end_index, "Chunk ")

        if self.embedding is None:
            object.__setattr__(self, "embedding", [])

    @classmethod
    def create(
        cls,
        document_id: UUID,
        library_id: UUID,
        text: str,
        embedding: list[float] | None = None,
        start_index: int = 0,
        end_index: int = 0,
        metadata: ChunkMetadata | None = None,
    ) -> "Chunk":
        """Create new Chunk with generated ID."""
        clean_text = validate_non_empty_text(text, "Chunk text cannot be empty")
        return cls(
            id=uuid.uuid4(),
            document_id=document_id,
            library_id=library_id,
            text=clean_text,
            embedding=embedding or [],
            start_index=start_index,
            end_index=end_index or len(clean_text),
            metadata=metadata or ChunkMetadata(),
        )

    def update(
        self,
        text: str | None = None,
        embedding: list[float] | None = None,
        start_index: int | None = None,
        end_index: int | None = None,
        metadata: ChunkMetadata | None = None,
    ) -> "Chunk":
        """Create new Chunk instance with updated fields.

        Raises:
            ValueError: If new text is empty or indices are invalid
        """
        updated_text = (
            validate_non_empty_text(text, "Chunk text cannot be empty")
            if text is not None
            else self.text
        )
        updated_start = (
            validate_non_negative(start_index, "Chunk start_index")
            if start_index is not None
            else self.start_index
        )
        updated_end = end_index if end_index is not None else self.end_index
        validate_index_range(updated_start, updated_end, "Chunk ")

        return Chunk(
            id=self.id,
            document_id=self.document_id,
            library_id=self.library_id,
            text=updated_text,
            embedding=embedding if embedding is not None else self.embedding,
            start_index=updated_start,
            end_index=updated_end,
            metadata=metadata if metadata is not None else self.metadata,
        )

    @property
    def has_embedding(self) -> bool:
        """Check if this chunk has a vector embedding."""
        return bool(self.embedding)

    @property
    def embedding_dim(self) -> int:
        """Get the dimensionality of the embedding vector."""
        return len(self.embedding) if self.embedding else 0
