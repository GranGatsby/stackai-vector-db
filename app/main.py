"""FastAPI application factory and main entry point."""

import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from time import perf_counter

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.errors import ERROR_HANDLERS
from app.api.v1.routers import (
    chunks_document_router,
    chunks_router,
    documents_library_router,
    documents_router,
    health_router,
    libraries_router,
)
from app.api.v1.routers.search import router as search_router
from app.core.config import settings
from app.core.logging import log_request_info, setup_logging


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager."""
    # Startup
    setup_logging()
    yield
    # Shutdown
    pass


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title=settings.api_title,
        description=settings.api_description,
        version=settings.api_version,
        debug=settings.debug,
        lifespan=lifespan,
    )

    # Add request logging middleware
    @app.middleware("http")
    async def request_logging_middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Log all HTTP requests with timing and unique request ID."""

        req_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        start_time = perf_counter()

        response: Response = await call_next(request)

        duration_ms = (perf_counter() - start_time) * 1000
        response.headers["x-request-id"] = req_id

        log_request_info(
            method=request.method,
            path=str(request.url.path),
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
            request_id=req_id,
        )

        return response

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )

    # Register error handlers
    for exception_class, handler in ERROR_HANDLERS.items():
        app.add_exception_handler(exception_class, handler)

    # Include routers
    app.include_router(health_router, prefix=settings.api_prefix)
    app.include_router(libraries_router, prefix=settings.api_prefix)
    app.include_router(documents_router, prefix=settings.api_prefix)
    app.include_router(documents_library_router, prefix=settings.api_prefix)
    app.include_router(chunks_router, prefix=settings.api_prefix)
    app.include_router(chunks_document_router, prefix=settings.api_prefix)
    app.include_router(search_router, prefix=settings.api_prefix)

    return app


# Create app instance
app = create_app()


def main() -> None:
    """Main entry point for running the application."""

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
