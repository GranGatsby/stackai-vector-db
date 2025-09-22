"""Unit tests for the InMemoryChunkRepository."""

import uuid

import pytest

from app.domain import Chunk, ChunkNotFoundError
from app.repositories.in_memory import InMemoryChunkRepository


class TestInMemoryChunkRepository:
    """Test suite for InMemoryChunkRepository."""

    @pytest.fixture
    def repository(self):
        """Create a fresh repository for each test."""
        return InMemoryChunkRepository()

    @pytest.fixture
    def library_id(self):
        """Create a sample library ID for testing."""
        return uuid.uuid4()

    @pytest.fixture
    def document_id(self):
        """Create a sample document ID for testing."""
        return uuid.uuid4()

    @pytest.fixture
    def sample_chunk(self, document_id, library_id):
        """Create a sample chunk for testing."""
        return Chunk.create(
            document_id=document_id,
            library_id=library_id,
            text="This is a test chunk",
            embedding=[0.1, 0.2, 0.3],
            start_index=0,
            end_index=20,
            metadata={"chunk_type": "paragraph"},
        )

    def test_chunk_repo_crud_roundtrip(
        self, repository: InMemoryChunkRepository, sample_chunk: Chunk
    ):
        """Test complete CRUD roundtrip for chunks."""
        # CREATE
        created = repository.create(sample_chunk)
        assert created == sample_chunk
        assert repository.exists(sample_chunk.id)
        assert repository.count_by_document(sample_chunk.document_id) == 1
        assert repository.count_by_library(sample_chunk.library_id) == 1

        # READ
        retrieved = repository.get_by_id(sample_chunk.id)
        assert retrieved == sample_chunk

        # UPDATE
        updated_chunk = sample_chunk.update(
            text="Updated text", embedding=[0.9, 0.8, 0.7]
        )
        result = repository.update(updated_chunk)
        assert result == updated_chunk
        assert repository.get_by_id(sample_chunk.id).text == "Updated text"

        # DELETE
        deleted = repository.delete(sample_chunk.id)
        assert deleted is True
        assert not repository.exists(sample_chunk.id)
        assert repository.get_by_id(sample_chunk.id) is None
        assert repository.count_by_document(sample_chunk.document_id) == 0
        assert repository.count_by_library(sample_chunk.library_id) == 0

    def test_list_chunks_by_document_with_pagination(
        self, repository: InMemoryChunkRepository, document_id, library_id
    ):
        """Test listing chunks by document with pagination."""
        # Create multiple chunks in the same document
        chunks = []
        for i in range(5):
            chunk = Chunk.create(
                document_id=document_id,
                library_id=library_id,
                text=f"Chunk {i} text",
                start_index=i * 10,
                end_index=(i * 10) + 9,
            )
            repository.create(chunk)
            chunks.append(chunk)

        # Test listing all chunks
        all_chunks = repository.list_by_document(document_id)
        assert len(all_chunks) == 5
        # Should be sorted by start_index
        assert all_chunks[0].start_index == 0
        assert all_chunks[-1].start_index == 40

        # Test pagination with limit
        limited = repository.list_by_document(document_id, limit=2)
        assert len(limited) == 2
        assert limited[0].start_index == 0
        assert limited[1].start_index == 10

        # Test pagination with offset
        offset_chunks = repository.list_by_document(document_id, offset=2)
        assert len(offset_chunks) == 3
        assert offset_chunks[0].start_index == 20

        # Test pagination with both limit and offset
        page_chunks = repository.list_by_document(document_id, limit=2, offset=1)
        assert len(page_chunks) == 2
        assert page_chunks[0].start_index == 10
        assert page_chunks[1].start_index == 20

        # Test count
        assert repository.count_by_document(document_id) == 5
        assert repository.count_by_library(library_id) == 5

    def test_list_chunks_by_library_with_pagination(
        self, repository: InMemoryChunkRepository, library_id
    ):
        """Test listing chunks by library with pagination."""
        doc1_id = uuid.uuid4()
        doc2_id = uuid.uuid4()

        # Create chunks in different documents but same library
        chunk1 = Chunk.create(
            document_id=doc1_id, library_id=library_id, text="Chunk 1", start_index=0, end_index=7
        )
        chunk2 = Chunk.create(
            document_id=doc1_id, library_id=library_id, text="Chunk 2", start_index=10, end_index=17
        )
        chunk3 = Chunk.create(
            document_id=doc2_id, library_id=library_id, text="Chunk 3", start_index=5, end_index=12
        )

        repository.create(chunk1)
        repository.create(chunk2)
        repository.create(chunk3)

        # Test listing all chunks by library
        all_chunks = repository.list_by_library(library_id)
        assert len(all_chunks) == 3
        # Should be sorted by document_id, then start_index
        assert all_chunks[0].start_index == 0  # doc1, start_index=0
        assert all_chunks[1].start_index == 10  # doc1, start_index=10

        # Test count by library
        assert repository.count_by_library(library_id) == 3

    def test_delete_chunk_updates_secondary_indexes(
        self, repository: InMemoryChunkRepository
    ):
        """Test that deleting chunks updates secondary indexes properly."""
        lib_id = uuid.uuid4()
        doc1_id = uuid.uuid4()
        doc2_id = uuid.uuid4()

        # Create chunks in different documents
        chunk1 = Chunk.create(
            document_id=doc1_id, library_id=lib_id, text="Chunk 1"
        )
        chunk2 = Chunk.create(
            document_id=doc1_id, library_id=lib_id, text="Chunk 2"
        )
        chunk3 = Chunk.create(
            document_id=doc2_id, library_id=lib_id, text="Chunk 3"
        )

        repository.create(chunk1)
        repository.create(chunk2)
        repository.create(chunk3)

        # Verify initial state
        assert repository.count_by_document(doc1_id) == 2
        assert repository.count_by_document(doc2_id) == 1
        assert repository.count_by_library(lib_id) == 3

        # Delete one chunk from doc1
        repository.delete(chunk1.id)

        # Verify indexes are updated
        assert repository.count_by_document(doc1_id) == 1
        assert repository.count_by_document(doc2_id) == 1
        assert repository.count_by_library(lib_id) == 2

        # Delete all chunks from doc1
        deleted_count = repository.delete_by_document(doc1_id)
        assert deleted_count == 1
        assert repository.count_by_document(doc1_id) == 0
        assert repository.count_by_document(doc2_id) == 1
        assert repository.count_by_library(lib_id) == 1

        # Delete all chunks from library
        deleted_count = repository.delete_by_library(lib_id)
        assert deleted_count == 1
        assert repository.count_by_library(lib_id) == 0

    def test_chunk_not_found_error(self, repository: InMemoryChunkRepository):
        """Test that updating non-existent chunk raises error."""
        non_existent_chunk = Chunk.create(
            document_id=uuid.uuid4(), library_id=uuid.uuid4(), text="Non-existent"
        )

        with pytest.raises(ChunkNotFoundError):
            repository.update(non_existent_chunk)

    def test_chunk_already_exists_error(
        self, repository: InMemoryChunkRepository, sample_chunk: Chunk
    ):
        """Test that creating duplicate chunk raises error."""
        repository.create(sample_chunk)

        with pytest.raises(ValueError, match="already exists"):
            repository.create(sample_chunk)

    def test_update_chunk_relationships_updates_indexes(
        self, repository: InMemoryChunkRepository
    ):
        """Test that updating chunk relationships updates secondary indexes."""
        old_doc_id = uuid.uuid4()
        new_doc_id = uuid.uuid4()
        old_lib_id = uuid.uuid4()
        new_lib_id = uuid.uuid4()

        # Create chunk in old document/library
        chunk = Chunk.create(
            document_id=old_doc_id, library_id=old_lib_id, text="Test Chunk"
        )
        repository.create(chunk)

        assert repository.count_by_document(old_doc_id) == 1
        assert repository.count_by_library(old_lib_id) == 1
        assert repository.count_by_document(new_doc_id) == 0
        assert repository.count_by_library(new_lib_id) == 0

        # Update chunk to new document/library
        updated_chunk = Chunk(
            id=chunk.id,
            document_id=new_doc_id,
            library_id=new_lib_id,
            text=chunk.text,
            embedding=chunk.embedding,
            start_index=chunk.start_index,
            end_index=chunk.end_index,
            metadata=chunk.metadata,
        )
        repository.update(updated_chunk)

        # Verify indexes are updated
        assert repository.count_by_document(old_doc_id) == 0
        assert repository.count_by_library(old_lib_id) == 0
        assert repository.count_by_document(new_doc_id) == 1
        assert repository.count_by_library(new_lib_id) == 1
