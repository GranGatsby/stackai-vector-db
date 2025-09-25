"""Unit tests for ChunkService."""

import uuid

import pytest

from app.clients.embedding import FakeEmbeddingClient
from app.domain import ChunkMetadata, ChunkNotFoundError, Document, Library
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
    def embedding_client(self):
        """Create a fake embedding client."""
        return FakeEmbeddingClient(embedding_dim=10)

    @pytest.fixture
    def service(self, chunk_repo, document_repo, library_repo, embedding_client):
        """Create a chunk service."""
        return ChunkService(chunk_repo, document_repo, library_repo, embedding_client)

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

    def test_create_chunks_success(
        self, service: ChunkService, sample_document: Document, sample_library: Library
    ):
        """Test successful chunk creation."""
        chunks_data = [{
            "text": "Test chunk content",
            "start_index": 0,
            "end_index": 18,
            "metadata": ChunkMetadata(chunk_type="paragraph"),
        }]
        chunks = service.create_chunks(
            document_id=sample_document.id,
            chunks_data=chunks_data,
        )

        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.document_id == sample_document.id
        assert chunk.library_id == sample_library.id
        assert chunk.text == "Test chunk content"
        assert chunk.start_index == 0
        assert chunk.end_index == 18
        assert chunk.metadata.chunk_type == "paragraph"
        assert service.chunk_exists(chunk.id)

    def test_create_chunks_with_embedding_computation(
        self, service: ChunkService, sample_document: Document, sample_library: Library
    ):
        """Test chunk creation with embedding computation."""
        chunks_data = [{
            "text": "Test chunk for embedding",
        }]
        chunks = service.create_chunks(
            document_id=sample_document.id,
            chunks_data=chunks_data,
            compute_embedding=True,
        )

        # Should have computed an embedding using FakeEmbeddingClient
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.has_embedding
        assert len(chunk.embedding) > 0
        # FakeEmbeddingClient uses fixed dimension of 10
        assert len(chunk.embedding) == 10

    def test_create_chunks_invalid_document(
        self, service: ChunkService, sample_library: Library
    ):
        """Test chunk creation with invalid document_id."""
        invalid_document_id = uuid.uuid4()
        chunks_data = [{"text": "Test chunk"}]

        with pytest.raises(ValueError, match="Document .* does not exist"):
            service.create_chunks(
                document_id=invalid_document_id,
                chunks_data=chunks_data,
            )

    def test_get_chunk_success(
        self, service: ChunkService, sample_document: Document, sample_library: Library
    ):
        """Test successful chunk retrieval."""
        chunks_data = [{"text": "Test chunk"}]
        created_chunks = service.create_chunks(
            document_id=sample_document.id,
            chunks_data=chunks_data,
        )
        created = created_chunks[0]

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
        chunks_data = [{"text": "Original text"}]
        created_chunks = service.create_chunks(
            document_id=sample_document.id,
            chunks_data=chunks_data,
        )
        chunk = created_chunks[0]

        updated = service.update_chunk(
            chunk.id,
            text="Updated text content",
            compute_embedding=True,
        )

        assert updated.text == "Updated text content"
        assert updated.has_embedding
        # Should have recomputed embedding using FakeEmbeddingClient
        assert len(updated.embedding) == 10

    def test_list_chunks_by_document_with_validation(
        self, service: ChunkService, sample_document: Document, sample_library: Library
    ):
        """Test listing chunks by document with validation."""
        # Create chunks
        chunks_data = [{"text": "Chunk 1"}, {"text": "Chunk 2"}]
        created_chunks = service.create_chunks(
            document_id=sample_document.id,
            chunks_data=chunks_data,
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
        chunks_data = [{"text": "Chunk 1"}, {"text": "Chunk 2"}]
        service.create_chunks(
            document_id=sample_document.id,
            chunks_data=chunks_data,
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
        chunks_data = [{"text": "Chunk 1"}, {"text": "Chunk 2"}]
        service.create_chunks(
            document_id=sample_document.id,
            chunks_data=chunks_data,
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

    def test_create_multiple_chunks_batch(
        self, service: ChunkService, sample_document: Document, sample_library: Library
    ):
        """Test creating multiple chunks in a single batch."""
        chunks_data = [
            {"text": "First chunk", "start_index": 0, "end_index": 11},
            {"text": "Second chunk", "start_index": 12, "end_index": 24},
            {"text": "Third chunk", "start_index": 25, "end_index": 36},
        ]
        
        chunks = service.create_chunks(
            document_id=sample_document.id,
            chunks_data=chunks_data,
            compute_embedding=True,
        )

        assert len(chunks) == 3
        for i, chunk in enumerate(chunks):
            assert chunk.text == chunks_data[i]["text"]
            assert chunk.start_index == chunks_data[i]["start_index"]
            assert chunk.end_index == chunks_data[i]["end_index"]
            assert chunk.has_embedding  # Should have embeddings computed
            assert len(chunk.embedding) == 10  # FakeEmbeddingClient dimension
