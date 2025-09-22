"""API schemas for the vector database.

This package contains Pydantic models for API request/response validation
and serialization. These schemas serve as the contract between the API
and its clients.
"""

from .chunk import ChunkCreate, ChunkList, ChunkRead, ChunkUpdate
from .document import DocumentCreate, DocumentList, DocumentRead, DocumentUpdate
from .errors import ErrorDetail, ErrorResponse
from .health import HealthResponse
from .library import LibraryBase, LibraryCreate, LibraryList, LibraryOut, LibraryUpdate

__all__ = [
    # Health
    "HealthResponse",
    # Library
    "LibraryBase",
    "LibraryCreate",
    "LibraryUpdate",
    "LibraryOut",
    "LibraryList",
    # Document
    "DocumentCreate",
    "DocumentUpdate",
    "DocumentRead",
    "DocumentList",
    # Chunk
    "ChunkCreate",
    "ChunkUpdate",
    "ChunkRead",
    "ChunkList",
    # Errors
    "ErrorDetail",
    "ErrorResponse",
]
