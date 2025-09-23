"""Unit tests for LibraryService cascade operations."""

import pytest

from app.repositories.in_memory import (
    InMemoryChunkRepository,
    InMemoryDocumentRepository,
    InMemoryLibraryRepository,
)
from app.services import ChunkService, DocumentService, LibraryService


class TestLibraryServiceCascade:
    """Test suite for LibraryService cascade operations."""

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
    def library_service(self, library_repo, document_repo, chunk_repo):
        """Create a library service with cascade support."""
        return LibraryService(library_repo, document_repo, chunk_repo)

    @pytest.fixture
    def document_service(self, document_repo, library_repo, chunk_repo):
        """Create a document service."""
        return DocumentService(document_repo, library_repo, chunk_repo)

    @pytest.fixture
    def embedding_client(self):
        """Create a fake embedding client."""
        from app.clients.embedding import FakeEmbeddingClient
        return FakeEmbeddingClient(embedding_dim=10)

    @pytest.fixture
    def chunk_service(self, chunk_repo, document_repo, library_repo, embedding_client):
        """Create a chunk service."""
        return ChunkService(chunk_repo, document_repo, library_repo, embedding_client)

    def test_delete_library_cascades_to_documents_and_chunks(
        self, library_service, document_service, chunk_service
    ):
        """Test that deleting a library cascades to documents and chunks."""
        # Create library
        library = library_service.create_library("Test Library")

        # Create documents
        doc1 = document_service.create_document(
            library_id=library.id, title="Document 1"
        )
        doc2 = document_service.create_document(
            library_id=library.id, title="Document 2"
        )

        # Create chunks
        chunk1 = chunk_service.create_chunk(
            document_id=doc1.id,
            library_id=library.id,
            text="Chunk 1",
            start_index=0,
            end_index=7,
        )
        chunk2 = chunk_service.create_chunk(
            document_id=doc1.id,
            library_id=library.id,
            text="Chunk 2",
            start_index=8,
            end_index=15,
        )
        chunk3 = chunk_service.create_chunk(
            document_id=doc2.id,
            library_id=library.id,
            text="Chunk 3",
            start_index=0,
            end_index=7,
        )

        # Verify everything exists
        assert library_service.library_exists(library.id)
        assert document_service.document_exists(doc1.id)
        assert document_service.document_exists(doc2.id)
        assert chunk_service.chunk_exists(chunk1.id)
        assert chunk_service.chunk_exists(chunk2.id)
        assert chunk_service.chunk_exists(chunk3.id)

        # Delete library (should cascade)
        deleted = library_service.delete_library(library.id)
        assert deleted is True

        # Verify library is deleted
        assert not library_service.library_exists(library.id)

        # Verify documents are deleted (cascaded)
        assert not document_service.document_exists(doc1.id)
        assert not document_service.document_exists(doc2.id)

        # Verify chunks are deleted (cascaded)
        assert not chunk_service.chunk_exists(chunk1.id)
        assert not chunk_service.chunk_exists(chunk2.id)
        assert not chunk_service.chunk_exists(chunk3.id)

    def test_delete_library_without_cascade_repos(self, library_repo):
        """Test library deletion when cascade repositories are not provided."""
        # Create service without cascade repositories
        service = LibraryService(library_repo)

        # Create library
        library = service.create_library("Test Library")
        assert service.library_exists(library.id)

        # Delete should still work (just without cascading)
        deleted = service.delete_library(library.id)
        assert deleted is True
        assert not service.library_exists(library.id)
