"""Chunk management endpoints."""

from uuid import UUID

from fastapi import APIRouter, Depends, Query, status

from app.api.v1.deps import get_chunk_service
from app.schemas import ChunkList, ChunkRead, ChunkUpdate
from app.schemas.chunk import ChunkCreateInDocument, ChunkMetadataSchema
from app.services import ChunkService

router = APIRouter(prefix="/chunks", tags=["chunks"])


@router.get(
    "/{chunk_id}",
    response_model=ChunkRead,
    status_code=status.HTTP_200_OK,
    summary="Get a chunk by ID",
    description="Retrieve a specific chunk by its unique identifier",
    responses={404: {"description": "Chunk not found"}},
)
async def get_chunk(
    chunk_id: UUID, service: ChunkService = Depends(get_chunk_service)
) -> ChunkRead:
    """Get a chunk by its ID."""
    chunk = service.get_chunk(chunk_id)
    return ChunkRead.from_domain(chunk)


@router.patch(
    "/{chunk_id}",
    response_model=ChunkRead,
    status_code=status.HTTP_200_OK,
    summary="Update a chunk",
    description="Update an existing chunk with new information",
    responses={
        404: {"description": "Chunk not found"},
        422: {"description": "Validation error"},
    },
)
async def update_chunk(
    chunk_id: UUID,
    chunk_data: ChunkUpdate,
    service: ChunkService = Depends(get_chunk_service),
) -> ChunkRead:
    """Update an existing chunk."""
    chunk = service.update_chunk(
        chunk_id=chunk_id,
        text=chunk_data.text,
        embedding=chunk_data.embedding,
        start_index=chunk_data.start_index,
        end_index=chunk_data.end_index,
        metadata=(
            ChunkMetadataSchema.dict_to_domain(chunk_data.metadata)
            if chunk_data.metadata is not None
            else None
        ),
        compute_embedding=chunk_data.compute_embedding,
    )
    return ChunkRead.from_domain(chunk)


@router.delete(
    "/{chunk_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a chunk",
    description="Delete a chunk by its unique identifier",
    responses={404: {"description": "Chunk not found"}},
)
async def delete_chunk(
    chunk_id: UUID, service: ChunkService = Depends(get_chunk_service)
) -> None:
    """Delete a chunk by its ID."""
    # First check if the chunk exists (will raise ChunkNotFoundError if not)
    service.get_chunk(chunk_id)

    # Then delete it
    service.delete_chunk(chunk_id)


# Document-scoped endpoints
document_router = APIRouter(prefix="/documents", tags=["chunks"])


@document_router.get(
    "/{document_id}/chunks",
    response_model=ChunkList,
    status_code=status.HTTP_200_OK,
    summary="List chunks in a document",
    description="Retrieve a paginated list of chunks in the specified document",
    responses={422: {"description": "Document not found"}},
)
async def list_chunks_by_document(
    document_id: UUID,
    limit: int | None = Query(
        None, ge=1, le=100, description="Maximum number of chunks to return"
    ),
    offset: int = Query(0, ge=0, description="Number of chunks to skip"),
    service: ChunkService = Depends(get_chunk_service),
) -> ChunkList:
    """List chunks in a document with optional pagination."""
    chunks, total = service.list_chunks_by_document(
        document_id, limit=limit, offset=offset
    )
    return ChunkList.from_domain_list(chunks, total=total, limit=limit, offset=offset)


@document_router.post(
    "/{document_id}/chunks",
    response_model=ChunkRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create a chunk in a document",
    description="Create a new chunk in the specified document with optional embedding computation",
    responses={
        422: {
            "description": "Validation error, document not found, or library mismatch"
        },
    },
)
async def create_chunk_in_document(
    document_id: UUID,
    chunk_data: ChunkCreateInDocument,
    service: ChunkService = Depends(get_chunk_service),
) -> ChunkRead:
    """Create a new chunk in a document."""
    # Override document_id from URL path
    chunk = service.create_chunk(
        document_id=document_id,
        library_id=chunk_data.library_id,
        text=chunk_data.text,
        embedding=chunk_data.embedding,
        start_index=chunk_data.start_index,
        end_index=chunk_data.end_index,
        metadata=ChunkMetadataSchema.dict_to_domain(chunk_data.metadata),
        compute_embedding=chunk_data.compute_embedding,
    )
    return ChunkRead.from_domain(chunk)
