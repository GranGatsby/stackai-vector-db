"""Unit tests for chunk schemas."""

import uuid

import pytest
from pydantic import ValidationError

from app.schemas import ChunkCreateInDocument, ChunkRead, ChunkUpdate
from app.schemas.chunk import ChunkMetadataSchema


class TestChunkSchemas:
    """Test suite for chunk schemas."""

    def test_chunk_create_valid(self):
        """Test valid chunk creation schema."""
        data = {
            "chunks": [
                {
                    "text": "Test chunk content",
                    "embedding": [0.1, 0.2, 0.3],
                    "start_index": 0,
                    "end_index": 18,
                    "metadata": {"chunk_type": "paragraph"},
                }
            ],
            "compute_embedding": True,
        }

        schema = ChunkCreateInDocument(**data)
        assert len(schema.chunks) == 1
        chunk = schema.chunks[0]
        assert chunk.text == "Test chunk content"
        assert chunk.embedding == [0.1, 0.2, 0.3]
        assert chunk.start_index == 0
        assert chunk.end_index == 18
        assert chunk.metadata.chunk_type == "paragraph"
        assert schema.compute_embedding is True

    def test_chunk_create_minimal(self):
        """Test chunk creation with minimal required fields."""
        data = {
            "chunks": [
                {
                    "text": "Minimal chunk",
                }
            ]
        }

        schema = ChunkCreateInDocument(**data)
        assert len(schema.chunks) == 1
        chunk = schema.chunks[0]
        assert chunk.text == "Minimal chunk"
        assert chunk.embedding == []  # Default
        assert chunk.start_index == 0  # Default
        assert chunk.end_index == 0  # Default
        assert isinstance(chunk.metadata, ChunkMetadataSchema)  # Default empty schema
        assert schema.compute_embedding is False  # Default

    def test_chunk_create_invalid_text(self):
        """Test chunk creation with invalid text."""

        # Empty text
        with pytest.raises(ValidationError, match="at least 1 character"):
            ChunkCreateInDocument(chunks=[{"text": ""}])

        # Whitespace only text
        with pytest.raises(ValidationError, match="empty or whitespace"):
            ChunkCreateInDocument(chunks=[{"text": "   "}])

    def test_chunk_create_invalid_indices(self):
        """Test chunk creation with invalid indices."""

        # end_index < start_index
        with pytest.raises(ValidationError, match="end_index must be >= start_index"):
            ChunkCreateInDocument(
                chunks=[
                    {
                        "text": "Test",
                        "start_index": 10,
                        "end_index": 5,
                    }
                ]
            )

        # Negative start_index
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            ChunkCreateInDocument(
                chunks=[
                    {
                        "text": "Test",
                        "start_index": -1,
                    }
                ]
            )

    def test_chunk_create_empty_chunks_list(self):
        """Test chunk creation with empty chunks list."""
        with pytest.raises(ValidationError, match="List should have at least 1 item"):
            ChunkCreateInDocument(chunks=[])

    def test_chunk_create_too_many_chunks(self):
        """Test chunk creation with too many chunks."""
        # Create 101 chunks (over the limit)
        chunks = [{"text": f"Chunk {i}"} for i in range(101)]
        with pytest.raises(ValidationError, match="List should have at most 100 items"):
            ChunkCreateInDocument(chunks=chunks)

    def test_chunk_create_multiple_chunks(self):
        """Test creating multiple chunks at once."""
        data = {
            "chunks": [
                {"text": "First chunk", "start_index": 0, "end_index": 11},
                {"text": "Second chunk", "start_index": 12, "end_index": 24},
                {"text": "Third chunk", "start_index": 25, "end_index": 36},
            ],
            "compute_embedding": True,
        }

        schema = ChunkCreateInDocument(**data)
        assert len(schema.chunks) == 3
        assert schema.chunks[0].text == "First chunk"
        assert schema.chunks[1].text == "Second chunk"
        assert schema.chunks[2].text == "Third chunk"
        assert schema.compute_embedding is True

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
        # metadata is now a ChunkMetadataSchema object
        assert isinstance(schema.metadata, ChunkMetadataSchema)
        assert schema.metadata.chunk_type == "test"
