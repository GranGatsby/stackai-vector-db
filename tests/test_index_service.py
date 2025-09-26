"""Unit tests for IndexService."""

import uuid
from unittest.mock import Mock

import pytest

from app.clients.embedding import FakeEmbeddingClient
from app.domain import (
    Chunk,
    ChunkMetadata,
    Library,
    LibraryNotFoundError,
    VectorIndexBuildError,
    VectorIndexNotBuiltError,
)
from app.repositories.in_memory import (
    InMemoryChunkRepository,
    InMemoryLibraryRepository,
)
from app.services.index_service import IndexAlgo, IndexService, IndexStatus


class TestIndexService:
    """Test suite for IndexService."""

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
    def service(self, library_repo, chunk_repo, embedding_client):
        """Create an index service."""
        repo, _ = library_repo
        return IndexService(repo, chunk_repo, embedding_client, rebuild_threshold=0.1)

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

    def test_get_status_library_not_found(self, service):
        """Test get_status fails when library doesn't exist."""
        non_existent_id = uuid.uuid4()

        with pytest.raises(LibraryNotFoundError):
            service.get_status(non_existent_id)

    def test_get_status_initial_state(self, service, library_repo):
        """Test get_status returns correct initial state."""
        _, library = library_repo

        status = service.get_status(library.id)

        assert isinstance(status, IndexStatus)
        assert status.library_id == library.id
        assert status.algorithm == IndexAlgo.LINEAR  # Default
        assert not status.is_built
        assert status.is_dirty
        assert status.size == 0
        assert status.embedding_dim is None
        assert status.built_at is None
        assert status.version == 0
        assert status.dirty_count == 0

    def test_build_index_success(self, service, library_repo, sample_chunks):
        """Test successful index building."""
        _, library = library_repo

        # Build the index
        status = service.build(library.id, IndexAlgo.LINEAR)

        # Verify build results
        assert status.is_built
        assert not status.is_dirty
        assert status.size == 3
        assert status.embedding_dim == 128
        assert status.version == 1
        assert status.built_at is not None
        assert status.algorithm == IndexAlgo.LINEAR

    def test_build_index_empty_library(self, service, library_repo):
        """Test building index on empty library creates empty index."""
        _, library = library_repo

        # Build should succeed but create empty index
        status = service.build(library.id)
        
        assert status.is_built
        assert status.size == 0
        assert status.embedding_dim == 1024  # Default from settings
        assert not status.is_dirty
        assert status.version == 1

    def test_mark_dirty_and_rebuild(self, service, library_repo, sample_chunks):
        """Test dirty flag management and rebuild triggering."""
        _, library = library_repo

        # Build initial index
        status = service.build(library.id)
        assert not status.is_dirty
        initial_version = status.version

        # Mark as dirty
        service.mark_dirty(library.id)
        status = service.get_status(library.id)
        assert status.is_dirty
        assert status.dirty_count == 1

        # Rebuild
        status = service.build(library.id)
        assert not status.is_dirty
        assert status.version == initial_version + 1
        assert status.dirty_count == 0

    def test_query_before_build_fails(self, service, library_repo, sample_chunks):
        """Test querying before building fails."""
        _, library = library_repo
        query_vector = [0.5] * 128

        with pytest.raises(VectorIndexNotBuiltError):
            service.query(library.id, query_vector, k=1)

    def test_query_after_build_success(self, service, library_repo, sample_chunks):
        """Test successful querying after building."""
        _, library = library_repo
        query_vector = [0.0, 1.0, 2.0] + [0.0] * 125  # Should match first chunk

        # Build index first
        service.build(library.id)

        # Query
        results = service.query(library.id, query_vector, k=2)

        assert len(results) <= 2
        assert all(isinstance(chunk_id, uuid.UUID) for chunk_id, _ in results)
        assert all(isinstance(distance, float) for _, distance in results)
        # Results should be sorted by distance (ascending)
        if len(results) > 1:
            assert results[0][1] <= results[1][1]
