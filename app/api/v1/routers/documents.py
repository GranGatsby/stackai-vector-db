"""Document management endpoints."""

from uuid import UUID

from fastapi import APIRouter, Depends, Query, status

from app.api.v1.deps import get_document_service
from app.schemas import DocumentList, DocumentRead, DocumentUpdate
from app.schemas.document import DocumentCreateInLibrary, DocumentMetadataSchema
from app.services import DocumentService

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get(
    "/{document_id}",
    response_model=DocumentRead,
    status_code=status.HTTP_200_OK,
    summary="Get a document by ID",
    description="Retrieve a specific document by its unique identifier",
    responses={status.HTTP_404_NOT_FOUND: {"description": "Document not found"}},
)
async def get_document(
    document_id: UUID, service: DocumentService = Depends(get_document_service)
) -> DocumentRead:
    """Get a document by its ID."""
    document = service.get_document(document_id)
    return DocumentRead.from_domain(document)


@router.patch(
    "/{document_id}",
    response_model=DocumentRead,
    status_code=status.HTTP_200_OK,
    summary="Update a document",
    description="Update an existing document with new information",
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Document not found"},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Validation error"},
    },
)
async def update_document(
    document_id: UUID,
    document_data: DocumentUpdate,
    service: DocumentService = Depends(get_document_service),
) -> DocumentRead:
    """Update an existing document."""
    document = service.update_document(
        document_id=document_id,
        title=document_data.title,
        content=document_data.content,
        metadata=(
            document_data.metadata.to_domain()
            if document_data.metadata is not None
            else None
        ),
    )
    return DocumentRead.from_domain(document)


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document",
    description="Delete a document by its unique identifier (cascades to chunks)",
    responses={status.HTTP_404_NOT_FOUND: {"description": "Document not found"}},
)
async def delete_document(
    document_id: UUID, service: DocumentService = Depends(get_document_service)
) -> None:
    """Delete a document by its ID."""
    # First check if the document exists (will raise DocumentNotFoundError if not)
    service.get_document(document_id)

    # Then delete it (cascades to chunks)
    service.delete_document(document_id)


# Library-scoped endpoints
library_router = APIRouter(prefix="/libraries", tags=["documents"])


@library_router.get(
    "/{library_id}/documents",
    response_model=DocumentList,
    status_code=status.HTTP_200_OK,
    summary="List documents in a library",
    description="Retrieve a paginated list of documents in the specified library",
    responses={status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Library not found"}},
)
async def list_documents_by_library(
    library_id: UUID,
    limit: int | None = Query(
        None, ge=1, le=100, description="Maximum number of documents to return"
    ),
    offset: int = Query(0, ge=0, description="Number of documents to skip"),
    service: DocumentService = Depends(get_document_service),
) -> DocumentList:
    """List documents in a library with optional pagination."""
    documents, total = service.list_documents_by_library(
        library_id, limit=limit, offset=offset
    )
    return DocumentList.from_domain_list(
        documents, total=total, limit=limit, offset=offset
    )


@library_router.post(
    "/{library_id}/documents",
    response_model=DocumentRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create a document in a library",
    description="Create a new document in the specified library",
    responses={
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Validation error or library not found"},
    },
)
async def create_document_in_library(
    library_id: UUID,
    document_data: DocumentCreateInLibrary,
    service: DocumentService = Depends(get_document_service),
) -> DocumentRead:
    """Create a new document in a library."""
    # Override library_id from URL path
    document = service.create_document(
        library_id=library_id,
        title=document_data.title,
        content=document_data.content,
        metadata=document_data.metadata.to_domain(),
    )
    return DocumentRead.from_domain(document)
