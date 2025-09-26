"""Library management endpoints."""

from uuid import UUID

from fastapi import APIRouter, Depends, Query, status

from app.api.v1.deps import get_library_service
from app.schemas import LibraryCreate, LibraryList, LibraryOut, LibraryUpdate
from app.schemas.library import LibraryMetadataSchema
from app.services import LibraryService

router = APIRouter(prefix="/libraries", tags=["libraries"])


@router.get(
    "/",
    response_model=LibraryList,
    status_code=status.HTTP_200_OK,
    summary="List all libraries",
    description="Retrieve a paginated list of libraries in the system",
)
async def list_libraries(
    limit: int | None = Query(
        None, ge=1, le=100, description="Maximum number of libraries to return"
    ),
    offset: int = Query(0, ge=0, description="Number of libraries to skip"),
    service: LibraryService = Depends(get_library_service),
) -> LibraryList:
    """List libraries with optional pagination."""
    libraries, total = service.list_libraries(limit=limit, offset=offset)
    return LibraryList.from_domain_list(
        libraries, total=total, limit=limit, offset=offset
    )


@router.get(
    "/{library_id}",
    response_model=LibraryOut,
    status_code=status.HTTP_200_OK,
    summary="Get a library by ID",
    description="Retrieve a specific library by its unique identifier",
    responses={status.HTTP_404_NOT_FOUND: {"description": "Library not found"}},
)
async def get_library(
    library_id: UUID, service: LibraryService = Depends(get_library_service)
) -> LibraryOut:
    """Get a library by its ID."""
    library = service.get_library(library_id)
    return LibraryOut.from_domain(library)


@router.post(
    "/",
    response_model=LibraryOut,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new library",
    description="Create a new library with the provided information",
    responses={
        status.HTTP_409_CONFLICT: {
            "description": "Library with the same name already exists"
        },
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Validation error"},
    },
)
async def create_library(
    library_data: LibraryCreate, service: LibraryService = Depends(get_library_service)
) -> LibraryOut:
    """Create a new library."""
    library = service.create_library(
        name=library_data.name,
        description=library_data.description,
        metadata=library_data.metadata.to_domain(),
    )
    return LibraryOut.from_domain(library)


@router.patch(
    "/{library_id}",
    response_model=LibraryOut,
    status_code=status.HTTP_200_OK,
    summary="Update a library",
    description="Update an existing library with new information",
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Library not found"},
        status.HTTP_409_CONFLICT: {
            "description": "Library name conflicts with existing library"
        },
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Validation error"},
    },
)
async def update_library(
    library_id: UUID,
    library_data: LibraryUpdate,
    service: LibraryService = Depends(get_library_service),
) -> LibraryOut:
    """Update an existing library."""
    library = service.update_library(
        library_id=library_id,
        name=library_data.name,
        description=library_data.description,
        metadata=(
            library_data.metadata.to_domain()
            if library_data.metadata is not None
            else None
        ),
    )
    return LibraryOut.from_domain(library)


@router.delete(
    "/{library_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a library",
    description="Delete a library by its unique identifier",
    responses={status.HTTP_404_NOT_FOUND: {"description": "Library not found"}},
)
async def delete_library(
    library_id: UUID, service: LibraryService = Depends(get_library_service)
) -> None:
    """Delete a library by its ID."""
    # First check if the library exists (will raise LibraryNotFoundError if not)
    service.get_library(library_id)

    # Then delete it
    service.delete_library(library_id)
