"""Unit tests for chunk schemas."""

import uuid

import pytest
from pydantic import ValidationError

from app.schemas import ChunkCreate, ChunkRead, ChunkUpdate


class TestChunkSchemas:
    """Test suite for chunk schemas."""

    def test_chunk_create_valid(self):
        """Test valid chunk creation schema."""
        document_id = uuid.uuid4()
        library_id = uuid.uuid4()
        data = {
            "document_id": document_id,
            "library_id": library_id,
            "text": "Test chunk content",
            "embedding": [0.1, 0.2, 0.3],
            "start_index": 0,
            "end_index": 18,
            "metadata": {"type": "paragraph"},
            "compute_embedding": True,
        }

        schema = ChunkCreate(**data)
        assert schema.document_id == document_id
        assert schema.library_id == library_id
        assert schema.text == "Test chunk content"
        assert schema.embedding == [0.1, 0.2, 0.3]
        assert schema.start_index == 0
        assert schema.end_index == 18
        assert schema.metadata["type"] == "paragraph"
        assert schema.compute_embedding is True

    def test_chunk_create_minimal(self):
        """Test chunk creation with minimal required fields."""
        document_id = uuid.uuid4()
        library_id = uuid.uuid4()
        data = {
            "document_id": document_id,
            "library_id": library_id,
            "text": "Minimal chunk",
        }

        schema = ChunkCreate(**data)
        assert schema.document_id == document_id
        assert schema.library_id == library_id
        assert schema.text == "Minimal chunk"
        assert schema.embedding == []  # Default
        assert schema.start_index == 0  # Default
        assert schema.end_index == 0  # Default
        assert schema.metadata == {}  # Default
        assert schema.compute_embedding is False  # Default

    def test_chunk_create_invalid_text(self):
        """Test chunk creation with invalid text."""
        document_id = uuid.uuid4()
        library_id = uuid.uuid4()

        # Empty text
        with pytest.raises(ValidationError, match="at least 1 character"):
            ChunkCreate(document_id=document_id, library_id=library_id, text="")

        # Whitespace only text
        with pytest.raises(ValidationError, match="empty or whitespace"):
            ChunkCreate(document_id=document_id, library_id=library_id, text="   ")

    def test_chunk_create_invalid_indices(self):
        """Test chunk creation with invalid indices."""
        document_id = uuid.uuid4()
        library_id = uuid.uuid4()

        # end_index < start_index
        with pytest.raises(ValidationError, match="end_index must be >= start_index"):
            ChunkCreate(
                document_id=document_id,
                library_id=library_id,
                text="Test",
                start_index=10,
                end_index=5,
            )

        # Negative start_index
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            ChunkCreate(
                document_id=document_id,
                library_id=library_id,
                text="Test",
                start_index=-1,
            )

    def test_chunk_update_partial(self):
        """Test partial chunk update schema."""
        data = {
            "text": "Updated text",
            "embedding": [0.9, 0.8, 0.7],
            "compute_embedding": True,
        }

        schema = ChunkUpdate(**data)
        assert schema.text == "Updated text"
        assert schema.embedding == [0.9, 0.8, 0.7]
        assert schema.compute_embedding is True
        assert schema.start_index is None  # Not provided
        assert schema.end_index is None  # Not provided
        assert schema.metadata is None  # Not provided

    def test_chunk_update_empty_optional(self):
        """Test chunk update with no fields provided."""
        schema = ChunkUpdate()
        assert schema.text is None
        assert schema.embedding is None
        assert schema.start_index is None
        assert schema.end_index is None
        assert schema.metadata is None
        assert schema.compute_embedding is False  # Default

    def test_chunk_update_invalid_text(self):
        """Test chunk update with invalid text."""
        # Empty text
        with pytest.raises(ValidationError, match="at least 1 character"):
            ChunkUpdate(text="")

        # Whitespace only text
        with pytest.raises(ValidationError, match="empty or whitespace"):
            ChunkUpdate(text="   ")

    def test_chunk_read_from_domain(self):
        """Test ChunkRead creation from domain entity."""
        from app.domain import Chunk

        document_id = uuid.uuid4()
        library_id = uuid.uuid4()
        from app.domain import ChunkMetadata

        chunk = Chunk.create(
            document_id=document_id,
            library_id=library_id,
            text="Test chunk",
            embedding=[0.1, 0.2],
            start_index=0,
            end_index=10,
            metadata=ChunkMetadata(chunk_type="test"),
        )

        schema = ChunkRead.from_domain(chunk)
        assert schema.id == chunk.id
        assert schema.document_id == chunk.document_id
        assert schema.library_id == chunk.library_id
        assert schema.text == chunk.text
        assert schema.embedding == chunk.embedding
        assert schema.start_index == chunk.start_index
        assert schema.end_index == chunk.end_index
        # metadata is converted to dict in schema response
        assert "chunk_type" in schema.metadata
        assert schema.metadata["chunk_type"] == "test"
