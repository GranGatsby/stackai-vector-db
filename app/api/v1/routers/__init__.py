"""API v1 routers."""

from .chunks import document_router as chunks_document_router
from .chunks import router as chunks_router
from .documents import library_router as documents_library_router
from .documents import router as documents_router
from .health import router as health_router
from .libraries import router as libraries_router
from .search import router as search_router

__all__ = [
    "chunks_document_router",
    "chunks_router",
    "documents_library_router",
    "documents_router",
    "health_router",
    "libraries_router",
    "search_router",
]
