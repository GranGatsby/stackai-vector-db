"""Unit tests for ChunkService."""

import uuid

import pytest

from app.domain import ChunkNotFoundError, Document, Library
from app.repositories.in_memory import (
    InMemoryChunkRepository,
    InMemoryDocumentRepository,
    InMemoryLibraryRepository,
)
from app.services import ChunkService


class TestChunkService:
    """Test suite for ChunkService."""

    @pytest.fixture
    def library_repo(self):
        """Create a library repository."""
        return InMemoryLibraryRepository()

    @pytest.fixture
    def document_repo(self):
        """Create a document repository."""
        return InMemoryDocumentRepository()

    @pytest.fixture
    def chunk_repo(self):
        """Create a chunk repository."""
        return InMemoryChunkRepository()

    @pytest.fixture
    def service(self, chunk_repo, document_repo, library_repo):
        """Create a chunk service."""
        return ChunkService(chunk_repo, document_repo, library_repo)

    @pytest.fixture
    def sample_library(self, library_repo):
        """Create a sample library."""
        library = Library.create(name="Test Library")
        library_repo.create(library)
        return library

    @pytest.fixture
    def sample_document(self, document_repo, sample_library):
        """Create a sample document."""
        document = Document.create(library_id=sample_library.id, title="Test Document")
        document_repo.create(document)
        return document

    def test_create_chunk_success(
        self, service: ChunkService, sample_document: Document, sample_library: Library
    ):
        """Test successful chunk creation."""
        chunk = service.create_chunk(
            document_id=sample_document.id,
            library_id=sample_library.id,
            text="Test chunk content",
            start_index=0,
            end_index=18,
            metadata={"type": "paragraph"},
        )

        assert chunk.document_id == sample_document.id
        assert chunk.library_id == sample_library.id
        assert chunk.text == "Test chunk content"
        assert chunk.start_index == 0
        assert chunk.end_index == 18
        assert chunk.metadata["type"] == "paragraph"
        assert service.chunk_exists(chunk.id)

    def test_create_chunk_with_embedding_placeholder(
        self, service: ChunkService, sample_document: Document, sample_library: Library
    ):
        """Test chunk creation with embedding computation."""
        chunk = service.create_chunk(
            document_id=sample_document.id,
            library_id=sample_library.id,
            text="Test chunk for embedding",
            compute_embedding=True,
        )

        # Should have computed a placeholder embedding
        assert chunk.has_embedding
        assert len(chunk.embedding) > 0
        # Placeholder embedding should be based on text length
        expected_dim = min(768, max(1, len("Test chunk for embedding") // 10))
        assert len(chunk.embedding) == expected_dim

    def test_create_chunk_invalid_document(
        self, service: ChunkService, sample_library: Library
    ):
        """Test chunk creation with invalid document_id."""
        invalid_document_id = uuid.uuid4()

        with pytest.raises(ValueError, match="Document .* does not exist"):
            service.create_chunk(
                document_id=invalid_document_id,
                library_id=sample_library.id,
                text="Test chunk",
            )

    def test_create_chunk_invalid_library(
        self, service: ChunkService, sample_document: Document
    ):
        """Test chunk creation with invalid library_id."""
        invalid_library_id = uuid.uuid4()

        with pytest.raises(ValueError, match="Library .* does not exist"):
            service.create_chunk(
                document_id=sample_document.id,
                library_id=invalid_library_id,
                text="Test chunk",
            )

    def test_create_chunk_document_library_mismatch(
        self, service: ChunkService, sample_document: Document, library_repo
    ):
        """Test chunk creation when document doesn't belong to specified library."""
        # Create another library
        other_library = Library.create(name="Other Library")
        library_repo.create(other_library)

        with pytest.raises(ValueError, match="does not belong to library"):
            service.create_chunk(
                document_id=sample_document.id,
                library_id=other_library.id,  # Wrong library!
                text="Test chunk",
            )

    def test_get_chunk_success(
        self, service: ChunkService, sample_document: Document, sample_library: Library
    ):
        """Test successful chunk retrieval."""
        created = service.create_chunk(
            document_id=sample_document.id,
            library_id=sample_library.id,
            text="Test chunk",
        )

        retrieved = service.get_chunk(created.id)
        assert retrieved == created

    def test_get_chunk_not_found(self, service: ChunkService):
        """Test chunk retrieval when chunk doesn't exist."""
        non_existent_id = uuid.uuid4()

        with pytest.raises(ChunkNotFoundError):
            service.get_chunk(non_existent_id)

    def test_update_chunk_with_embedding_recompute(
        self, service: ChunkService, sample_document: Document, sample_library: Library
    ):
        """Test chunk update with embedding recomputation."""
        chunk = service.create_chunk(
            document_id=sample_document.id,
            library_id=sample_library.id,
            text="Original text",
        )

        updated = service.update_chunk(
            chunk.id,
            text="Updated text content",
            compute_embedding=True,
        )

        assert updated.text == "Updated text content"
        assert updated.has_embedding
        # Should have recomputed embedding for new text
        expected_dim = min(768, max(1, len("Updated text content") // 10))
        assert len(updated.embedding) == expected_dim

    def test_list_chunks_by_document_with_validation(
        self, service: ChunkService, sample_document: Document, sample_library: Library
    ):
        """Test listing chunks by document with validation."""
        # Create chunks
        chunk1 = service.create_chunk(
            document_id=sample_document.id,
            library_id=sample_library.id,
            text="Chunk 1",
        )
        chunk2 = service.create_chunk(
            document_id=sample_document.id,
            library_id=sample_library.id,
            text="Chunk 2",
        )

        # List chunks
        chunks, total = service.list_chunks_by_document(sample_document.id)
        assert len(chunks) == 2
        assert total == 2

        # Test with invalid document
        invalid_document_id = uuid.uuid4()
        with pytest.raises(ValueError, match="Document .* does not exist"):
            service.list_chunks_by_document(invalid_document_id)

    def test_list_chunks_by_library_with_validation(
        self, service: ChunkService, sample_document: Document, sample_library: Library
    ):
        """Test listing chunks by library with validation."""
        # Create chunks
        service.create_chunk(
            document_id=sample_document.id,
            library_id=sample_library.id,
            text="Chunk 1",
        )
        service.create_chunk(
            document_id=sample_document.id,
            library_id=sample_library.id,
            text="Chunk 2",
        )

        # List chunks
        chunks, total = service.list_chunks_by_library(sample_library.id)
        assert len(chunks) == 2
        assert total == 2

        # Test with invalid library
        invalid_library_id = uuid.uuid4()
        with pytest.raises(ValueError, match="Library .* does not exist"):
            service.list_chunks_by_library(invalid_library_id)

    def test_count_chunks_validation(
        self, service: ChunkService, sample_document: Document, sample_library: Library
    ):
        """Test chunk counting with validation."""
        # Initially no chunks
        assert service.count_chunks_by_document(sample_document.id) == 0
        assert service.count_chunks_by_library(sample_library.id) == 0

        # Create chunks
        service.create_chunk(
            document_id=sample_document.id,
            library_id=sample_library.id,
            text="Chunk 1",
        )
        service.create_chunk(
            document_id=sample_document.id,
            library_id=sample_library.id,
            text="Chunk 2",
        )

        # Counts should be updated
        assert service.count_chunks_by_document(sample_document.id) == 2
        assert service.count_chunks_by_library(sample_library.id) == 2

        # Test with invalid IDs
        invalid_document_id = uuid.uuid4()
        invalid_library_id = uuid.uuid4()

        with pytest.raises(ValueError, match="Document .* does not exist"):
            service.count_chunks_by_document(invalid_document_id)

        with pytest.raises(ValueError, match="Library .* does not exist"):
            service.count_chunks_by_library(invalid_library_id)
