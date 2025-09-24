"""Error handlers for mapping domain errors to HTTP responses.

This module contains exception handlers that translate domain-specific
exceptions into appropriate HTTP responses with consistent error formatting.
"""

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.domain import (
    ChunkNotFoundError,
    DocumentAlreadyExistsError,
    DocumentNotFoundError,
    DomainError,
    EmbeddingDimensionMismatchError,
    EmptyLibraryError,
    IndexBuildError,
    IndexNotBuiltError,
    InvalidSearchParameterError,
    LibraryAlreadyExistsError,
    LibraryNotFoundError,
    SearchError,
)
from app.domain import (
    ValidationError as DomainValidationError,
)
from app.schemas import ErrorResponse


def _create_error_response(
    status_code: int,
    error_code: str,
    message: str,
    field: str = None,
    context: dict = None,
) -> JSONResponse:
    """Helper to create consistent error responses.

    Args:
        status_code: HTTP status code
        error_code: Error code string
        message: Error message
        field: Field name if error is field-specific
        context: Additional error context

    Returns:
        JSONResponse with consistent error format
    """
    error_response = ErrorResponse(
        error={
            "code": error_code,
            "message": message,
            "field": field,
            "context": context,
        }
    )
    return JSONResponse(status_code=status_code, content=error_response.model_dump())


def _create_not_found_response(resource_type: str, identifier: str) -> JSONResponse:
    """Helper to create 404 not found responses."""
    return _create_error_response(
        status_code=status.HTTP_404_NOT_FOUND,
        error_code="NOT_FOUND",
        message=f"{resource_type} with identifier '{identifier}' not found",
    )


def _create_conflict_response(message: str) -> JSONResponse:
    """Helper to create 409 conflict responses."""
    return _create_error_response(
        status_code=status.HTTP_409_CONFLICT, error_code="CONFLICT", message=message
    )


def _create_validation_response(message: str, field: str = None) -> JSONResponse:
    """Helper to create 422 validation error responses."""
    return _create_error_response(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        error_code="VALIDATION_ERROR",
        message=message,
        field=field,
    )


async def library_not_found_handler(
    request: Request, exc: LibraryNotFoundError
) -> JSONResponse:
    """Handle LibraryNotFoundError exceptions."""
    return _create_not_found_response("Library", exc.library_id)


async def library_already_exists_handler(
    request: Request, exc: LibraryAlreadyExistsError
) -> JSONResponse:
    """Handle LibraryAlreadyExistsError exceptions."""
    return _create_conflict_response(f"Library with name '{exc.name}' already exists")


async def document_not_found_handler(
    request: Request, exc: DocumentNotFoundError
) -> JSONResponse:
    """Handle DocumentNotFoundError exceptions."""
    return _create_not_found_response("Document", exc.document_id)


async def document_already_exists_handler(
    request: Request, exc: DocumentAlreadyExistsError
) -> JSONResponse:
    """Handle DocumentAlreadyExistsError exceptions."""
    return _create_conflict_response(
        f"Document with title '{exc.title}' already exists in library '{exc.library_id}'"
    )


async def chunk_not_found_handler(
    request: Request, exc: ChunkNotFoundError
) -> JSONResponse:
    """Handle ChunkNotFoundError exceptions."""
    return _create_not_found_response("Chunk", exc.chunk_id)


async def index_not_built_handler(
    request: Request, exc: IndexNotBuiltError
) -> JSONResponse:
    """Handle IndexNotBuiltError exceptions."""
    return _create_error_response(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        error_code="INDEX_NOT_BUILT",
        message=f"Index for library '{exc.library_id}' has not been built. Please build the index first.",
    )


async def index_build_error_handler(
    request: Request, exc: IndexBuildError
) -> JSONResponse:
    """Handle IndexBuildError exceptions."""
    return _create_error_response(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        error_code="INDEX_BUILD_FAILED",
        message=f"Failed to build index for library '{exc.library_id}': {exc.reason}",
    )


async def embedding_dimension_mismatch_handler(
    request: Request, exc: EmbeddingDimensionMismatchError
) -> JSONResponse:
    """Handle EmbeddingDimensionMismatchError exceptions."""
    return _create_error_response(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        error_code="EMBEDDING_DIMENSION_MISMATCH",
        message=f"Embedding dimension mismatch: expected {exc.expected}, got {exc.actual}",
    )


async def domain_validation_error_handler(
    request: Request, exc: DomainValidationError
) -> JSONResponse:
    """Handle domain ValidationError exceptions."""
    return _create_validation_response(str(exc))


async def pydantic_validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle Pydantic validation errors."""
    # Extract the first error for simplicity
    first_error = exc.errors()[0] if exc.errors() else {}
    field = ".".join(str(loc) for loc in first_error.get("loc", []))
    message = first_error.get("msg", "Validation error")

    return _create_validation_response(message, field if field else None)


async def generic_domain_error_handler(
    request: Request, exc: DomainError
) -> JSONResponse:
    """Handle generic domain errors."""
    return _create_error_response(
        status_code=status.HTTP_400_BAD_REQUEST,
        error_code=getattr(exc, "code", exc.__class__.__name__),
        message=str(exc),
    )


async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """Handle ValueError exceptions (often from domain validation)."""
    return _create_validation_response(str(exc))


async def empty_library_error_handler(
    request: Request, exc: EmptyLibraryError
) -> JSONResponse:
    """Handle EmptyLibraryError exceptions."""
    return _create_error_response(
        status_code=status.HTTP_409_CONFLICT,
        error_code="EMPTY_LIBRARY",
        message=f"Library '{exc.library_id}' is empty and cannot be searched",
    )


async def invalid_search_parameter_error_handler(
    request: Request, exc: InvalidSearchParameterError
) -> JSONResponse:
    """Handle InvalidSearchParameterError exceptions."""
    return _create_error_response(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        error_code="INVALID_SEARCH_PARAMETER",
        message=f"Invalid search parameter '{exc.parameter}': {exc.reason}",
        field=exc.parameter,
        context={"value": str(exc.value)},
    )


async def search_error_handler(request: Request, exc: SearchError) -> JSONResponse:
    """Handle generic SearchError exceptions."""
    return _create_error_response(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        error_code="SEARCH_ERROR",
        message=str(exc),
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    return _create_error_response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code="INTERNAL_ERROR",
        message="An unexpected error occurred",
    )


# Error handler registry for easy registration
ERROR_HANDLERS = {
    LibraryNotFoundError: library_not_found_handler,
    LibraryAlreadyExistsError: library_already_exists_handler,
    DocumentNotFoundError: document_not_found_handler,
    DocumentAlreadyExistsError: document_already_exists_handler,
    ChunkNotFoundError: chunk_not_found_handler,
    IndexNotBuiltError: index_not_built_handler,
    IndexBuildError: index_build_error_handler,
    EmbeddingDimensionMismatchError: embedding_dimension_mismatch_handler,
    EmptyLibraryError: empty_library_error_handler,
    InvalidSearchParameterError: invalid_search_parameter_error_handler,
    SearchError: search_error_handler,
    DomainValidationError: domain_validation_error_handler,
    RequestValidationError: pydantic_validation_error_handler,
    DomainError: generic_domain_error_handler,
    ValueError: value_error_handler,
    Exception: generic_exception_handler,
}
