"""Search and indexing endpoints."""

from uuid import UUID

from fastapi import APIRouter, Depends, status

from app.api.v1.deps import get_index_service, get_search_service
from app.core.config import settings
from app.domain import Chunk
from app.schemas import (
    BuildIndexRequest,
    BuildIndexResponse,
    IndexStatus,
    SearchByTextRequest,
    SearchByVectorRequest,
    SearchHit,
    SearchResult,
)
from app.schemas.chunk import ChunkMetadataSchema
from app.services import IndexService, SearchService
from app.services.index_service import IndexAlgo

router = APIRouter(prefix="/libraries", tags=["search"])




@router.post(
    "/{library_id}/index",
    response_model=BuildIndexResponse,
    status_code=status.HTTP_200_OK,
    summary="Build or rebuild library index",
    description="Build or rebuild the vector index for a library. This operation may take time for large libraries.",
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Library not found"},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Index build failed or validation error"},
    },
)
async def build_index(
    library_id: UUID,
    request: BuildIndexRequest,
    index_service: IndexService = Depends(get_index_service),
) -> BuildIndexResponse:
    """Build or rebuild the vector index for a library."""
    # Convert algorithm string to enum if provided
    algorithm_enum = None
    if request.algorithm is not None:
        algorithm_enum = IndexAlgo(request.algorithm)

    # Build the index with optional algorithm specification
    index_status = index_service.build(library_id, algorithm_enum)

    # Convert domain result to response schema
    return BuildIndexResponse.from_index_status(index_status, index_status.build_duration)


@router.get(
    "/{library_id}/index/status",
    response_model=IndexStatus,
    status_code=status.HTTP_200_OK,
    summary="Get library index status",
    description="Get detailed information about the current state of a library's vector index, including build status and dirty flags.",
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Library not found"},
    },
)
async def get_index_status(
    library_id: UUID,
    index_service: IndexService = Depends(get_index_service),
) -> IndexStatus:
    """Get the current status of a library's vector index."""
    # Get index status from service
    index_status = index_service.get_status(library_id)
    
    # Convert to schema (IndexStatus from service matches IndexStatus schema)
    return IndexStatus(
        library_id=index_status.library_id,
        algorithm=index_status.algorithm.value,
        is_built=index_status.is_built,
        is_dirty=index_status.is_dirty,
        size=index_status.size,
        embedding_dim=index_status.embedding_dim,
        built_at=index_status.built_at,
        version=index_status.version,
        dirty_count=index_status.dirty_count,
    )


@router.post(
    "/{library_id}/query/text",
    response_model=SearchResult,
    status_code=status.HTTP_200_OK,
    summary="Search by text query",
    description="Perform similarity search using text query. Generates embedding automatically. Results are filtered by each chunk's similarity_threshold if configured.",
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Library not found"},
        status.HTTP_409_CONFLICT: {"description": "Index not built or dirty"},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Invalid query parameters or validation error"},
    },
)
async def search_by_text(
    library_id: UUID,
    request: SearchByTextRequest,
    search_service: SearchService = Depends(get_search_service),
) -> SearchResult:
    """Search for similar chunks using text query."""
    # Execute text-based search
    search_result = search_service.query_text(library_id, request.text, request.k)

    # Convert domain result to response schema
    hits = [
        SearchHit(
            chunk_id=chunk.id,
            score=distance,
            metadata=ChunkMetadataSchema.from_domain(chunk.metadata) if chunk.metadata else None,
        )
        for chunk, distance in search_result.matches
    ]

    return SearchResult(
        hits=hits,
        total=search_result.total_results,
        library_id=search_result.library_id,
        algorithm=search_result.algorithm.value,
        index_size=search_result.index_size,
        embedding_dim=search_result.embedding_dim,
        query_embedding=search_result.query_embedding,
    )


@router.post(
    "/{library_id}/query/vector",
    response_model=SearchResult,
    status_code=status.HTTP_200_OK,
    summary="Search by embedding vector",
    description="Perform similarity search using pre-computed embedding vector. Results are filtered by each chunk's similarity_threshold if configured.",
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Library not found"},
        status.HTTP_409_CONFLICT: {"description": "Index not built or dirty"},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "description": "Invalid query parameters, dimension mismatch, or validation error"
        },
    },
)
async def search_by_vector(
    library_id: UUID,
    request: SearchByVectorRequest,
    search_service: SearchService = Depends(get_search_service),
) -> SearchResult:
    """Search for similar chunks using embedding vector."""
    # Execute embedding-based search
    search_result = search_service.query_embedding(
        library_id, request.embedding, request.k
    )

    # Convert domain result to response schema
    hits = [
        SearchHit(
            chunk_id=chunk.id,
            score=distance,
            metadata=ChunkMetadataSchema.from_domain(chunk.metadata) if chunk.metadata else None,
        )
        for chunk, distance in search_result.matches
    ]

    return SearchResult(
        hits=hits,
        total=search_result.total_results,
        library_id=search_result.library_id,
        algorithm=search_result.algorithm.value,
        index_size=search_result.index_size,
        embedding_dim=search_result.embedding_dim,
        query_embedding=search_result.query_embedding,
    )
