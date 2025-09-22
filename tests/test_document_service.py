"""Unit tests for DocumentService."""

import uuid

import pytest

from app.domain import DocumentNotFoundError, Library
from app.repositories.in_memory import (
    InMemoryChunkRepository,
    InMemoryDocumentRepository,
    InMemoryLibraryRepository,
)
from app.services import DocumentService


class TestDocumentService:
    """Test suite for DocumentService."""

    @pytest.fixture
    def library_repo(self):
        """Create a library repository with test data."""
        repo = InMemoryLibraryRepository()
        return repo

    @pytest.fixture
    def document_repo(self):
        """Create a document repository."""
        return InMemoryDocumentRepository()

    @pytest.fixture
    def chunk_repo(self):
        """Create a chunk repository."""
        return InMemoryChunkRepository()

    @pytest.fixture
    def service(self, document_repo, library_repo, chunk_repo):
        """Create a document service."""
        return DocumentService(document_repo, library_repo, chunk_repo)

    @pytest.fixture
    def sample_library(self, library_repo):
        """Create a sample library."""
        library = Library.create(name="Test Library")
        library_repo.create(library)
        return library

    def test_create_document_success(
        self, service: DocumentService, sample_library: Library
    ):
        """Test successful document creation."""
        document = service.create_document(
            library_id=sample_library.id,
            title="Test Document",
            content="Test content",
            metadata={"author": "Test Author"},
        )

        assert document.library_id == sample_library.id
        assert document.title == "Test Document"
        assert document.content == "Test content"
        assert document.metadata["author"] == "Test Author"
        assert service.document_exists(document.id)

    def test_create_document_invalid_library(self, service: DocumentService):
        """Test document creation with invalid library_id."""
        invalid_library_id = uuid.uuid4()

        with pytest.raises(ValueError, match="does not exist"):
            service.create_document(
                library_id=invalid_library_id,
                title="Test Document",
            )

    def test_get_document_success(
        self, service: DocumentService, sample_library: Library
    ):
        """Test successful document retrieval."""
        created = service.create_document(
            library_id=sample_library.id, title="Test Document"
        )

        retrieved = service.get_document(created.id)
        assert retrieved == created

    def test_get_document_not_found(self, service: DocumentService):
        """Test document retrieval when document doesn't exist."""
        non_existent_id = uuid.uuid4()

        with pytest.raises(DocumentNotFoundError):
            service.get_document(non_existent_id)

    def test_update_document_success(
        self, service: DocumentService, sample_library: Library
    ):
        """Test successful document update."""
        document = service.create_document(
            library_id=sample_library.id, title="Original Title"
        )

        updated = service.update_document(
            document.id, title="Updated Title", content="New content"
        )

        assert updated.id == document.id
        assert updated.title == "Updated Title"
        assert updated.content == "New content"
        assert updated.library_id == sample_library.id

    def test_delete_document_with_cascade(
        self, service: DocumentService, sample_library: Library
    ):
        """Test document deletion with cascading chunk deletion."""
        from app.services import ChunkService

        # Create document
        document = service.create_document(
            library_id=sample_library.id, title="Test Document"
        )

        # Create chunks for the document using ChunkService
        chunk_service = ChunkService(
            service._chunk_repository,
            service._document_repository,
            service._library_repository,
        )

        chunk1 = chunk_service.create_chunk(
            document_id=document.id,
            library_id=sample_library.id,
            text="Chunk 1",
            start_index=0,
            end_index=7,
        )
        chunk2 = chunk_service.create_chunk(
            document_id=document.id,
            library_id=sample_library.id,
            text="Chunk 2",
            start_index=8,
            end_index=15,
        )

        # Verify chunks exist
        assert chunk_service.chunk_exists(chunk1.id)
        assert chunk_service.chunk_exists(chunk2.id)
        assert chunk_service.count_chunks_by_document(document.id) == 2

        # Delete document (should cascade to chunks)
        deleted = service.delete_document(document.id)
        assert deleted is True

        # Verify document is deleted
        assert not service.document_exists(document.id)

        # Verify chunks are deleted (cascaded)
        assert not chunk_service.chunk_exists(chunk1.id)
        assert not chunk_service.chunk_exists(chunk2.id)
        # Note: Can't count chunks by document after document is deleted
        # The chunks should be gone from the chunk repository directly
        assert len(service._chunk_repository.list_by_document(document.id)) == 0

    def test_list_documents_by_library_with_validation(
        self, service: DocumentService, sample_library: Library
    ):
        """Test listing documents with library validation."""
        # Create some documents
        doc1 = service.create_document(library_id=sample_library.id, title="Doc 1")
        doc2 = service.create_document(library_id=sample_library.id, title="Doc 2")

        # List documents
        documents, total = service.list_documents_by_library(sample_library.id)
        assert len(documents) == 2
        assert total == 2

        # Test with invalid library
        invalid_library_id = uuid.uuid4()
        with pytest.raises(ValueError, match="does not exist"):
            service.list_documents_by_library(invalid_library_id)

    def test_count_documents_by_library_with_validation(
        self, service: DocumentService, sample_library: Library
    ):
        """Test counting documents with library validation."""
        # Initially no documents
        count = service.count_documents_by_library(sample_library.id)
        assert count == 0

        # Create documents
        service.create_document(library_id=sample_library.id, title="Doc 1")
        service.create_document(library_id=sample_library.id, title="Doc 2")

        # Count should be updated
        count = service.count_documents_by_library(sample_library.id)
        assert count == 2

        # Test with invalid library
        invalid_library_id = uuid.uuid4()
        with pytest.raises(ValueError, match="does not exist"):
            service.count_documents_by_library(invalid_library_id)
