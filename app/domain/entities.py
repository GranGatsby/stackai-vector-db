"""Domain entities for the vector database.

This module contains the core business entities that represent the fundamental
concepts of the vector database: Library, Document, and Chunk.
"""

from dataclasses import dataclass
from typing import Dict, Any
from uuid import UUID
import uuid


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
        metadata: Additional key-value metadata for the library
    """
    id: UUID
    name: str
    description: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self) -> None:
        """Validate library invariants after initialization."""
        if not self.name or not self.name.strip():
            raise ValueError("Library name cannot be empty")
        
        if len(self.name.strip()) > 255:
            raise ValueError("Library name cannot exceed 255 characters")
        
        # Ensure metadata is never None for consistency
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})
    
    @classmethod
    def create(
        cls,
        name: str,
        description: str = "",
        metadata: Dict[str, Any] = None
    ) -> "Library":
        """Factory method to create a new Library with generated ID.
        
        Args:
            name: The library name
            description: Optional description
            metadata: Optional metadata dictionary
            
        Returns:
            A new Library instance with a generated UUID
            
        Raises:
            ValueError: If name is empty or too long
        """
        return cls(
            id=uuid.uuid4(),
            name=name.strip(),
            description=description,
            metadata=metadata or {}
        )
    
    def update(
        self,
        name: str = None,
        description: str = None,
        metadata: Dict[str, Any] = None
    ) -> "Library":
        """Create a new Library instance with updated fields.
        
        Since Library is immutable, this returns a new instance with
        the specified fields updated.
        
        Args:
            name: New name (if provided)
            description: New description (if provided)  
            metadata: New metadata (if provided)
            
        Returns:
            A new Library instance with updated fields
            
        Raises:
            ValueError: If new name is empty or too long
        """
        return Library(
            id=self.id,
            name=name.strip() if name is not None else self.name,
            description=description if description is not None else self.description,
            metadata=metadata if metadata is not None else self.metadata
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
    metadata: Dict[str, Any] = None
    
    def __post_init__(self) -> None:
        """Validate document invariants."""
        if not self.title or not self.title.strip():
            raise ValueError("Document title cannot be empty")
            
        if len(self.title.strip()) > 255:
            raise ValueError("Document title cannot exceed 255 characters")
            
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})
    
    @classmethod
    def create(
        cls,
        library_id: UUID,
        title: str,
        content: str = "",
        metadata: Dict[str, Any] = None
    ) -> "Document":
        """Factory method to create a new Document with generated ID."""
        return cls(
            id=uuid.uuid4(),
            library_id=library_id,
            title=title.strip(),
            content=content,
            metadata=metadata or {}
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
    metadata: Dict[str, Any] = None
    
    def __post_init__(self) -> None:
        """Validate chunk invariants."""
        if not self.text or not self.text.strip():
            raise ValueError("Chunk text cannot be empty")
            
        if self.start_index < 0:
            raise ValueError("Chunk start_index cannot be negative")
            
        if self.end_index < self.start_index:
            raise ValueError("Chunk end_index must be >= start_index")
            
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})
            
        if self.embedding is None:
            object.__setattr__(self, 'embedding', [])
    
    @classmethod
    def create(
        cls,
        document_id: UUID,
        library_id: UUID,
        text: str,
        embedding: list[float] = None,
        start_index: int = 0,
        end_index: int = 0,
        metadata: Dict[str, Any] = None
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
            metadata=metadata or {}
        )
    
    @property
    def has_embedding(self) -> bool:
        """Check if this chunk has a vector embedding."""
        return bool(self.embedding)
    
    @property
    def embedding_dim(self) -> int:
        """Get the dimensionality of the embedding vector."""
        return len(self.embedding) if self.embedding else 0
