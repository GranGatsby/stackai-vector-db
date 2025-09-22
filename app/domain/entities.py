"""Domain entities for the vector database.

This module contains the core business entities that represent the fundamental
concepts of the vector database: Library, Document, and Chunk.
"""

import uuid
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import UUID


@dataclass(frozen=True)
class LibraryMetadata:
    """Fixed metadata schema for Library entities.
    
    Provides a structured schema for library metadata instead of dict[str, Any].
    This follows the task requirement for fixed schemas to reduce validation complexity.
    """
    
    author: Optional[str] = None
    version: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    created_by: Optional[str] = None
    project: Optional[str] = None
    category: Optional[str] = None
    is_public: bool = True
    # Test/workflow fields
    test: Optional[bool] = None
    updated: Optional[bool] = None
    original: Optional[bool] = None
    workflow: Optional[str] = None


@dataclass(frozen=True)
class Library:
    """A collection of documents with associated metadata.

    A Library serves as a container for documents and provides organizational
    structure for vector search operations. It maintains immutable state to
    ensure consistency across concurrent operations.

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
        """Validate library invariants after initialization."""
        if not self.name or not self.name.strip():
            raise ValueError("Library name cannot be empty")

        if len(self.name.strip()) > 255:
            raise ValueError("Library name cannot exceed 255 characters")

    @classmethod
    def create(
        cls, name: str, description: str = "", metadata: Optional[LibraryMetadata] = None
    ) -> "Library":
        """Factory method to create a new Library with generated ID.

        Args:
            name: The library name
            description: Optional description
            metadata: Optional structured metadata

        Returns:
            A new Library instance with a generated UUID

        Raises:
            ValueError: If name is empty or too long
        """
        return cls(
            id=uuid.uuid4(),
            name=name.strip(),
            description=description,
            metadata=metadata or LibraryMetadata(),
        )

    def update(
        self, name: str = None, description: str = None, metadata: Optional[LibraryMetadata] = None
    ) -> "Library":
        """Create a new Library instance with updated fields.

        Since Library is immutable, this returns a new instance with
        the specified fields updated.

        Args:
            name: New name (if provided)
            description: New description (if provided)
            metadata: New structured metadata (if provided)

        Returns:
            A new Library instance with updated fields

        Raises:
            ValueError: If new name is empty or too long
        """
        return Library(
            id=self.id,
            name=name.strip() if name is not None else self.name,
            description=description if description is not None else self.description,
            metadata=metadata if metadata is not None else self.metadata,
        )


@dataclass(frozen=True)
class Document:
    """A document within a library, composed of multiple chunks.

    Documents represent logical units of content that can be indexed
    and searched. Each document belongs to a library and contains
    metadata about the source content.

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
    metadata: dict[str, Any] = None

    def __post_init__(self) -> None:
        """Validate document invariants."""
        if not self.title or not self.title.strip():
            raise ValueError("Document title cannot be empty")

        if len(self.title.strip()) > 255:
            raise ValueError("Document title cannot exceed 255 characters")

        if self.metadata is None:
            object.__setattr__(self, "metadata", {})

    @classmethod
    def create(
        cls,
        library_id: UUID,
        title: str,
        content: str = "",
        metadata: dict[str, Any] = None,
    ) -> "Document":
        """Factory method to create a new Document with generated ID."""
        return cls(
            id=uuid.uuid4(),
            library_id=library_id,
            title=title.strip(),
            content=content,
            metadata=metadata or {},
        )

    def update(
        self,
        title: str = None,
        content: str = None,
        metadata: dict[str, Any] = None,
    ) -> "Document":
        """Create a new Document instance with updated fields.

        Since Document is immutable, this returns a new instance with
        the specified fields updated.

        Args:
            title: New title (if provided)
            content: New content (if provided)
            metadata: New metadata (if provided)

        Returns:
            A new Document instance with updated fields

        Raises:
            ValueError: If new title is empty or too long
        """
        return Document(
            id=self.id,
            library_id=self.library_id,
            title=title.strip() if title is not None else self.title,
            content=content if content is not None else self.content,
            metadata=metadata if metadata is not None else self.metadata,
        )


@dataclass(frozen=True)
class Chunk:
    """A chunk of text with associated embedding and metadata.

    Chunks are the fundamental units for vector search operations.
    Each chunk contains text content, its vector embedding, and
    references to its parent document and library.

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
    embedding: list[float] = None
    start_index: int = 0
    end_index: int = 0
    metadata: dict[str, Any] = None

    def __post_init__(self) -> None:
        """Validate chunk invariants."""
        if not self.text or not self.text.strip():
            raise ValueError("Chunk text cannot be empty")

        if self.start_index < 0:
            raise ValueError("Chunk start_index cannot be negative")

        if self.end_index < self.start_index:
            raise ValueError("Chunk end_index must be >= start_index")

        if self.metadata is None:
            object.__setattr__(self, "metadata", {})

        if self.embedding is None:
            object.__setattr__(self, "embedding", [])

    @classmethod
    def create(
        cls,
        document_id: UUID,
        library_id: UUID,
        text: str,
        embedding: list[float] = None,
        start_index: int = 0,
        end_index: int = 0,
        metadata: dict[str, Any] = None,
    ) -> "Chunk":
        """Factory method to create a new Chunk with generated ID."""
        return cls(
            id=uuid.uuid4(),
            document_id=document_id,
            library_id=library_id,
            text=text.strip(),
            embedding=embedding or [],
            start_index=start_index,
            end_index=end_index or len(text.strip()),
            metadata=metadata or {},
        )

    def update(
        self,
        text: str = None,
        embedding: list[float] = None,
        start_index: int = None,
        end_index: int = None,
        metadata: dict[str, Any] = None,
    ) -> "Chunk":
        """Create a new Chunk instance with updated fields.

        Since Chunk is immutable, this returns a new instance with
        the specified fields updated.

        Args:
            text: New text content (if provided)
            embedding: New embedding vector (if provided)
            start_index: New start index (if provided)
            end_index: New end index (if provided)
            metadata: New metadata (if provided)

        Returns:
            A new Chunk instance with updated fields

        Raises:
            ValueError: If new text is empty or indices are invalid
        """
        return Chunk(
            id=self.id,
            document_id=self.document_id,
            library_id=self.library_id,
            text=text.strip() if text is not None else self.text,
            embedding=embedding if embedding is not None else self.embedding,
            start_index=start_index if start_index is not None else self.start_index,
            end_index=end_index if end_index is not None else self.end_index,
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
