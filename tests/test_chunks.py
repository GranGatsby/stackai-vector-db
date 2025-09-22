"""Unit tests for Chunk entity."""

import uuid
from uuid import UUID

import pytest

from app.domain import Chunk, Document, Library


class TestChunk:
    """Test suite for Chunk entity."""

    @pytest.fixture
    def document_id(self) -> UUID:
        """Create a sample document ID for testing."""
        return uuid.uuid4()

    @pytest.fixture
    def library_id(self) -> UUID:
        """Create a sample library ID for testing."""
        return uuid.uuid4()

    @pytest.fixture
    def sample_chunk(self, document_id: UUID, library_id: UUID) -> Chunk:
        """Create a sample chunk for testing."""
        from app.domain import ChunkMetadata

        return Chunk.create(
            document_id=document_id,
            library_id=library_id,
            text="This is a test chunk",
            embedding=[0.1, 0.2, 0.3],
            start_index=0,
            end_index=20,
            metadata=ChunkMetadata(chunk_type="paragraph"),
        )

    def test_chunk_create_success(self, document_id: UUID, library_id: UUID):
        """Test successful chunk creation."""
        from app.domain import ChunkMetadata

        chunk = Chunk.create(
            document_id=document_id,
            library_id=library_id,
            text="Test chunk text",
            embedding=[1.0, 2.0, 3.0],
            start_index=10,
            end_index=25,
            metadata=ChunkMetadata(tags=["test"]),
        )

        assert isinstance(chunk.id, UUID)
        assert chunk.document_id == document_id
        assert chunk.library_id == library_id
        assert chunk.text == "Test chunk text"
        assert chunk.embedding == [1.0, 2.0, 3.0]
        assert chunk.start_index == 10
        assert chunk.end_index == 25
        assert chunk.metadata == ChunkMetadata(tags=["test"])

    def test_chunk_create_minimal(self, document_id: UUID, library_id: UUID):
        """Test chunk creation with minimal required data."""
        from app.domain import ChunkMetadata

        chunk = Chunk.create(
            document_id=document_id,
            library_id=library_id,
            text="Minimal chunk",
        )

        assert isinstance(chunk.id, UUID)
        assert chunk.document_id == document_id
        assert chunk.library_id == library_id
        assert chunk.text == "Minimal chunk"
        assert chunk.embedding == []
        assert chunk.start_index == 0
        assert chunk.end_index == len("Minimal chunk")
        assert chunk.metadata == ChunkMetadata()

    def test_chunk_requires_document_id_and_non_empty_text(self, library_id: UUID):
        """Test that chunk requires document_id and non-empty text."""
        document_id = uuid.uuid4()

        # Should succeed with valid data
        chunk = Chunk.create(
            document_id=document_id,
            library_id=library_id,
            text="Valid text",
        )
        assert chunk.document_id == document_id
        assert chunk.text == "Valid text"

        # Should fail with empty text
        with pytest.raises(ValueError, match="Chunk text cannot be empty"):
            Chunk.create(
                document_id=document_id,
                library_id=library_id,
                text="",
            )

        # Should fail with whitespace-only text
        with pytest.raises(ValueError, match="Chunk text cannot be empty"):
            Chunk.create(
                document_id=document_id,
                library_id=library_id,
                text="   ",
            )

    def test_chunk_text_strips_whitespace(self, document_id: UUID, library_id: UUID):
        """Test that chunk text is stripped of whitespace."""
        chunk = Chunk.create(
            document_id=document_id,
            library_id=library_id,
            text="  Test chunk text  ",
        )
        assert chunk.text == "Test chunk text"

    def test_chunk_start_index_validation(self, document_id: UUID, library_id: UUID):
        """Test chunk start_index validation."""
        # Should fail with negative start_index
        with pytest.raises(ValueError, match="Chunk start_index cannot be negative"):
            Chunk.create(
                document_id=document_id,
                library_id=library_id,
                text="Test text",
                start_index=-1,
            )

    def test_chunk_end_index_validation(self, document_id: UUID, library_id: UUID):
        """Test chunk end_index validation."""
        # Should fail when end_index < start_index
        with pytest.raises(ValueError, match="Chunk end_index must be >= start_index"):
            Chunk.create(
                document_id=document_id,
                library_id=library_id,
                text="Test text",
                start_index=10,
                end_index=5,
            )

    def test_chunk_embedding_default_empty_list(
        self, document_id: UUID, library_id: UUID
    ):
        """Test that embedding defaults to empty list."""
        chunk = Chunk.create(
            document_id=document_id,
            library_id=library_id,
            text="Test text",
        )
        assert chunk.embedding == []

    def test_chunk_metadata_default_empty_dict(
        self, document_id: UUID, library_id: UUID
    ):
        """Test that metadata defaults to empty ChunkMetadata."""
        from app.domain import ChunkMetadata

        chunk = Chunk.create(
            document_id=document_id,
            library_id=library_id,
            text="Test text",
        )
        assert chunk.metadata == ChunkMetadata()

    def test_chunk_has_embedding_property(self, document_id: UUID, library_id: UUID):
        """Test has_embedding property."""
        # Chunk without embedding
        chunk_no_embedding = Chunk.create(
            document_id=document_id,
            library_id=library_id,
            text="Test text",
        )
        assert not chunk_no_embedding.has_embedding

        # Chunk with embedding
        chunk_with_embedding = Chunk.create(
            document_id=document_id,
            library_id=library_id,
            text="Test text",
            embedding=[0.1, 0.2, 0.3],
        )
        assert chunk_with_embedding.has_embedding

    def test_chunk_embedding_dim_property(self, document_id: UUID, library_id: UUID):
        """Test embedding_dim property."""
        # Chunk without embedding
        chunk_no_embedding = Chunk.create(
            document_id=document_id,
            library_id=library_id,
            text="Test text",
        )
        assert chunk_no_embedding.embedding_dim == 0

        # Chunk with embedding
        chunk_with_embedding = Chunk.create(
            document_id=document_id,
            library_id=library_id,
            text="Test text",
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
        )
        assert chunk_with_embedding.embedding_dim == 5

    def test_chunk_update_success(self, sample_chunk: Chunk):
        """Test successful chunk update."""
        from app.domain import ChunkMetadata

        updated_chunk = sample_chunk.update(
            text="Updated text",
            embedding=[0.9, 0.8, 0.7],
            start_index=5,
            end_index=17,
            metadata=ChunkMetadata(tags=["updated"]),
        )

        # Original chunk unchanged
        assert sample_chunk.text == "This is a test chunk"
        assert sample_chunk.embedding == [0.1, 0.2, 0.3]
        assert sample_chunk.start_index == 0
        assert sample_chunk.end_index == 20
        assert sample_chunk.metadata == ChunkMetadata(chunk_type="paragraph")

        # New chunk has updated values
        assert updated_chunk.id == sample_chunk.id  # Same ID
        assert updated_chunk.document_id == sample_chunk.document_id  # Same document_id
        assert updated_chunk.library_id == sample_chunk.library_id  # Same library_id
        assert updated_chunk.text == "Updated text"
        assert updated_chunk.embedding == [0.9, 0.8, 0.7]
        assert updated_chunk.start_index == 5
        assert updated_chunk.end_index == 17
        assert updated_chunk.metadata == ChunkMetadata(tags=["updated"])

    def test_chunk_update_partial(self, sample_chunk: Chunk):
        """Test partial chunk update."""
        updated_chunk = sample_chunk.update(text="New text only")

        # Only text should change
        assert updated_chunk.text == "New text only"
        assert updated_chunk.embedding == sample_chunk.embedding
        assert updated_chunk.start_index == sample_chunk.start_index
        assert updated_chunk.end_index == sample_chunk.end_index
        assert updated_chunk.metadata == sample_chunk.metadata

    def test_chunk_update_returns_new_instance(self, sample_chunk: Chunk):
        """Test that update returns a new instance and keeps old unchanged."""
        updated_chunk = sample_chunk.update(text="New text")

        # Different objects
        assert updated_chunk is not sample_chunk
        assert id(updated_chunk) != id(sample_chunk)

        # Original unchanged
        assert sample_chunk.text == "This is a test chunk"

    def test_chunk_immutable(self, sample_chunk: Chunk):
        """Test that chunk is immutable (frozen dataclass)."""
        with pytest.raises(AttributeError):
            sample_chunk.text = "Cannot change"

    def test_chunk_validation_preserves_ids(self, sample_chunk: Chunk):
        """Test that update preserves ID, document_id, and library_id."""
        updated_chunk = sample_chunk.update(text="New text")

        assert updated_chunk.id == sample_chunk.id
        assert updated_chunk.document_id == sample_chunk.document_id
        assert updated_chunk.library_id == sample_chunk.library_id

    def test_chunk_update_returns_new_instance_and_keeps_old_unchanged(self):
        """Test that Chunk.update() returns new instance and keeps old unchanged."""
        document_id = uuid.uuid4()
        library_id = uuid.uuid4()
        from app.domain import ChunkMetadata

        original_chunk = Chunk.create(
            document_id=document_id,
            library_id=library_id,
            text="Original text",
            embedding=[0.1, 0.2, 0.3],
            start_index=0,
            end_index=13,
            metadata=ChunkMetadata(tags=["original"]),
        )

        updated_chunk = original_chunk.update(
            text="Updated text",
            embedding=[0.9, 0.8, 0.7],
            start_index=5,
            end_index=17,
            metadata=ChunkMetadata(tags=["updated"]),
        )

        # Different instances
        assert updated_chunk is not original_chunk
        assert id(updated_chunk) != id(original_chunk)

        # Original unchanged
        assert original_chunk.text == "Original text"
        assert original_chunk.embedding == [0.1, 0.2, 0.3]
        assert original_chunk.start_index == 0
        assert original_chunk.end_index == 13
        assert original_chunk.metadata == ChunkMetadata(tags=["original"])

        # Updated has new values
        assert updated_chunk.text == "Updated text"
        assert updated_chunk.embedding == [0.9, 0.8, 0.7]
        assert updated_chunk.start_index == 5
        assert updated_chunk.end_index == 17
        assert updated_chunk.metadata == ChunkMetadata(tags=["updated"])

        # Same ID, document_id, and library_id
        assert updated_chunk.id == original_chunk.id
        assert updated_chunk.document_id == original_chunk.document_id
        assert updated_chunk.library_id == original_chunk.library_id


class TestChunkIntegrationWithDocumentAndLibrary:
    """Test suite for Chunk integration with Document and Library."""

    def test_chunk_can_be_created_with_document_and_library_references(self):
        """Test that Chunk can reference existing Document and Library."""
        # Create a library first
        library = Library.create(
            name="Test Library",
            description="A library for testing chunks",
        )

        # Create a document in the library
        document = Document.create(
            library_id=library.id,
            title="Test Document",
            content="This is a test document with some content to chunk",
        )

        # Create a chunk that references both document and library
        chunk = Chunk.create(
            document_id=document.id,
            library_id=library.id,
            text="This is a test document",
            start_index=0,
            end_index=24,
        )

        assert chunk.document_id == document.id
        assert chunk.library_id == library.id
        assert chunk.library_id == document.library_id  # Consistency check
        assert isinstance(chunk.id, UUID)
        assert chunk.id != document.id  # Different entities have different IDs
        assert chunk.id != library.id

    def test_chunk_preserves_relationships_on_update(self):
        """Test that Chunk maintains relationships after updates."""
        library = Library.create(name="Persistent Library")
        document = Document.create(
            library_id=library.id,
            title="Persistent Document",
        )

        original_chunk = Chunk.create(
            document_id=document.id,
            library_id=library.id,
            text="Original chunk text",
        )

        updated_chunk = original_chunk.update(
            text="Updated chunk text",
            embedding=[0.1, 0.2, 0.3],
        )

        # Relationships preserved
        assert updated_chunk.document_id == document.id
        assert updated_chunk.library_id == library.id
        assert updated_chunk.document_id == original_chunk.document_id
        assert updated_chunk.library_id == original_chunk.library_id

    def test_chunk_text_indices_consistency(self):
        """Test that chunk indices are consistent with text content."""
        document_id = uuid.uuid4()
        library_id = uuid.uuid4()

        full_text = "This is a sample document with multiple sentences for chunking."
        chunk_text = "sample document"
        start_idx = full_text.index(chunk_text)
        end_idx = start_idx + len(chunk_text)

        chunk = Chunk.create(
            document_id=document_id,
            library_id=library_id,
            text=chunk_text,
            start_index=start_idx,
            end_index=end_idx,
        )

        assert chunk.text == chunk_text
        assert chunk.start_index == start_idx
        assert chunk.end_index == end_idx
        assert chunk.end_index - chunk.start_index == len(chunk_text)

    def test_chunk_embedding_operations(self):
        """Test chunk embedding-related operations."""
        document_id = uuid.uuid4()
        library_id = uuid.uuid4()

        # Test chunk without embedding
        chunk_no_embedding = Chunk.create(
            document_id=document_id,
            library_id=library_id,
            text="Text without embedding",
        )

        assert not chunk_no_embedding.has_embedding
        assert chunk_no_embedding.embedding_dim == 0
        assert chunk_no_embedding.embedding == []

        # Test chunk with embedding
        embedding_vector = [0.1, 0.2, 0.3, 0.4, 0.5]
        chunk_with_embedding = Chunk.create(
            document_id=document_id,
            library_id=library_id,
            text="Text with embedding",
            embedding=embedding_vector,
        )

        assert chunk_with_embedding.has_embedding
        assert chunk_with_embedding.embedding_dim == len(embedding_vector)
        assert chunk_with_embedding.embedding == embedding_vector

        # Test updating embedding
        new_embedding = [0.9, 0.8, 0.7]
        updated_chunk = chunk_with_embedding.update(embedding=new_embedding)

        assert updated_chunk.has_embedding
        assert updated_chunk.embedding_dim == len(new_embedding)
        assert updated_chunk.embedding == new_embedding

        # Original unchanged
        assert chunk_with_embedding.embedding == embedding_vector
