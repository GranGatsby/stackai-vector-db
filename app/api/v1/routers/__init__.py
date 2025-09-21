"""API v1 routers.

This package contains all the API endpoint routers for version 1 of the API.
Each router handles a specific domain area (health, libraries, documents, etc.).
"""

from .health import router as health_router
from .libraries import router as libraries_router

__all__ = [
    "health_router",
    "libraries_router",
]
