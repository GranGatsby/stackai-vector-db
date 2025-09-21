"""API schemas for the vector database.

This package contains Pydantic models for API request/response validation
and serialization. These schemas serve as the contract between the API
and its clients.
"""

from .health import HealthResponse
from .library import LibraryBase, LibraryCreate, LibraryUpdate, LibraryOut, LibraryList
from .errors import ErrorDetail, ErrorResponse

__all__ = [
    # Health
    "HealthResponse",
    # Library
    "LibraryBase",
    "LibraryCreate",
    "LibraryUpdate",
    "LibraryOut",
    "LibraryList",
    # Errors
    "ErrorDetail",
    "ErrorResponse",
]
