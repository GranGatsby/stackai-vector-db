"""API schemas for the vector database.

This package contains Pydantic models for API request/response validation
and serialization. These schemas serve as the contract between the API
and its clients.
"""

from .chunk import (
    ChunkCreateInDocument,
    ChunkCreateResponse,
    ChunkList,
    ChunkMetadataSchema,
    ChunkRead,
    ChunkUpdate,
)
from .document import (
    DocumentCreateInLibrary,
    DocumentList,
    DocumentRead,
    DocumentUpdate,
)
from .errors import ErrorDetail, ErrorResponse
from .health import HealthResponse
from .index import BuildIndexRequest, BuildIndexResponse, IndexAlgo, IndexStatus
from .library import LibraryBase, LibraryCreate, LibraryList, LibraryOut, LibraryUpdate
from .search import SearchByTextRequest, SearchByVectorRequest, SearchHit, SearchResult

__all__ = [
    # Index
    "BuildIndexRequest",
    "BuildIndexResponse",
    # Chunk
    "ChunkCreateInDocument",
    "ChunkCreateResponse",
    "ChunkList",
    "ChunkMetadataSchema",
    "ChunkRead",
    "ChunkUpdate",
    # Document
    "DocumentCreateInLibrary",
    "DocumentList",
    "DocumentRead",
    "DocumentUpdate",
    # Errors
    "ErrorDetail",
    "ErrorResponse",
    # Health
    "HealthResponse",
    # Index
    "IndexAlgo",
    "IndexStatus",
    # Library
    "LibraryBase",
    "LibraryCreate",
    "LibraryList",
    "LibraryOut",
    "LibraryUpdate",
    # Search
    "SearchByTextRequest",
    "SearchByVectorRequest",
    "SearchHit",
    "SearchResult",
]
