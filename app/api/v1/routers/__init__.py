"""API v1 routers.

This package contains all the API endpoint routers for version 1 of the API.
Each router handles a specific domain area (health, libraries, documents, etc.).
"""

from .chunks import document_router as chunks_document_router
from .chunks import router as chunks_router
from .documents import library_router as documents_library_router
from .documents import router as documents_router
from .health import router as health_router
from .libraries import router as libraries_router

__all__ = [
    "health_router",
    "libraries_router",
    "documents_router",
    "documents_library_router",
    "chunks_router",
    "chunks_document_router",
]
