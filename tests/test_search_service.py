"""Unit tests for SearchService."""

import uuid
from unittest.mock import Mock, patch

import pytest

from app.clients.embedding import EmbeddingResult, FakeEmbeddingClient
from app.domain import (
    Chunk,
    ChunkMetadata,
    EmbeddingDimensionMismatchError,
    EmptyLibraryError,
    InvalidSearchParameterError,
    Library,
    LibraryNotFoundError,
    VectorIndexNotBuiltError,
)
from app.repositories.in_memory import (
    InMemoryChunkRepository,
    InMemoryLibraryRepository,
)
from app.services.index_service import IndexAlgo, IndexService, IndexStatus
from app.services.search_service import SearchResult, SearchService


class TestSearchService:
    """Test suite for SearchService."""

    @pytest.fixture
    def library_repo(self):
        """Create a library repository with test data."""
        repo = InMemoryLibraryRepository()
        library = Library.create(name="Test Library")
        repo.create(library)
        return repo, library

    @pytest.fixture
    def chunk_repo(self):
        """Create a chunk repository."""
        return InMemoryChunkRepository()

    @pytest.fixture
    def embedding_client(self):
        """Create a fake embedding client."""
        return FakeEmbeddingClient(embedding_dim=128)

    @pytest.fixture
    def mock_index_service(self):
        """Create a mock index service."""
        return Mock(spec=IndexService)

    @pytest.fixture
    def service(self, library_repo, chunk_repo, embedding_client, mock_index_service):
        """Create a search service."""
        repo, _ = library_repo
        return SearchService(mock_index_service, repo, chunk_repo, embedding_client)

    @pytest.fixture
    def sample_chunks(self, library_repo, chunk_repo):
        """Create sample chunks with embeddings."""
        _, library = library_repo
        doc_id = uuid.uuid4()
        
        chunks = []
        for i in range(3):
            chunk = Chunk.create(
                document_id=doc_id,
                library_id=library.id,
                text=f"Test chunk {i}",
                embedding=[float(i), float(i+1), float(i+2)] + [0.0] * 125,  # 128-dim
                start_index=i * 10,
                end_index=(i * 10) + 9,
                metadata=ChunkMetadata(chunk_type="test"),
            )
            chunk_repo.create(chunk)
            chunks.append(chunk)
        return chunks

    def test_query_text_success(self, service, library_repo, mock_index_service, sample_chunks):
        """Test successful text query with embedding generation."""
        _, library = library_repo
        
        # Mock index service responses
        index_status = IndexStatus(
            library_id=library.id,
            algorithm=IndexAlgo.LINEAR,
            is_built=True,
            is_dirty=False,
            size=3,
            embedding_dim=128,
            built_at=1234567890.0,
            version=1,
            dirty_count=0,
        )
        mock_index_service.get_status.return_value = index_status
        mock_index_service.query.return_value = [
            (sample_chunks[0].id, 0.1),
            (sample_chunks[1].id, 0.2),
        ]

        result = service.query_text(library.id, "test query", k=2)

        # Verify result
        assert isinstance(result, SearchResult)
        assert result.library_id == library.id
        assert result.total_results == 2
        assert len(result.matches) == 2
        assert result.algorithm == IndexAlgo.LINEAR
        assert result.embedding_dim == 128

        # Verify embedding was generated and used
        assert len(result.query_embedding) == 128
        mock_index_service.query.assert_called_once()

    def test_query_embedding_success(self, service, library_repo, mock_index_service, sample_chunks):
        """Test successful embedding query."""
        _, library = library_repo
        query_embedding = [0.5] * 128

        # Mock index service responses
        index_status = IndexStatus(
            library_id=library.id,
            algorithm=IndexAlgo.LINEAR,
            is_built=True,
            is_dirty=False,
            size=3,
            embedding_dim=128,
            built_at=1234567890.0,
            version=1,
            dirty_count=0,
        )
        mock_index_service.get_status.return_value = index_status
        mock_index_service.query.return_value = [(sample_chunks[0].id, 0.1)]

        result = service.query_embedding(library.id, query_embedding, k=1)

        assert result.library_id == library.id
        assert result.total_results == 1
        assert result.query_embedding == query_embedding
        mock_index_service.query.assert_called_once_with(library.id, query_embedding, 1)

    def test_query_library_not_found(self, service, mock_index_service):
        """Test query fails when library doesn't exist."""
        non_existent_id = uuid.uuid4()

        with pytest.raises(LibraryNotFoundError):
            service.query_text(non_existent_id, "test query")

    def test_query_index_not_built(self, service, library_repo, mock_index_service):
        """Test query fails when index is not built and library is empty."""
        _, library = library_repo

        # Mock index service to indicate index not built initially
        initial_status = IndexStatus(
            library_id=library.id,
            algorithm=IndexAlgo.LINEAR,
            is_built=False,
            is_dirty=True,
            size=0,
            embedding_dim=None,
            built_at=None,
            version=0,
            dirty_count=0,
        )
        
        # After build, it should still be empty (no chunks)
        built_status = IndexStatus(
            library_id=library.id,
            algorithm=IndexAlgo.LINEAR,
            is_built=True,
            is_dirty=False,
            size=0,  # Empty library
            embedding_dim=128,
            built_at=1234567890.0,
            version=1,
            dirty_count=0,
        )
        
        mock_index_service.get_status.return_value = initial_status
        mock_index_service.build.return_value = built_status

        # Should fail with EmptyLibraryError, not VectorIndexNotBuiltError
        from app.domain import EmptyLibraryError
        with pytest.raises(EmptyLibraryError):
            service.query_text(library.id, "test query")

    def test_query_dimension_mismatch(self, service, library_repo, mock_index_service, sample_chunks):
        """Test query fails with dimension mismatch."""
        _, library = library_repo
        wrong_embedding = [0.5] * 64  # Wrong dimension

        # Mock index service responses
        index_status = IndexStatus(
            library_id=library.id,
            algorithm=IndexAlgo.LINEAR,
            is_built=True,
            is_dirty=False,
            size=3,
            embedding_dim=128,  # Expected 128, but query has 64
            built_at=1234567890.0,
            version=1,
            dirty_count=0,
        )
        mock_index_service.get_status.return_value = index_status

        with pytest.raises(EmbeddingDimensionMismatchError):
            service.query_embedding(library.id, wrong_embedding)
