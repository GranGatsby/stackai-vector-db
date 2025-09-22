"""Unit tests for Document entity."""

import uuid
from uuid import UUID

import pytest

from app.domain import Document, Library


class TestDocument:
    """Test suite for Document entity."""

    @pytest.fixture
    def library_id(self) -> UUID:
        """Create a sample library ID for testing."""
        return uuid.uuid4()

    @pytest.fixture
    def sample_document(self, library_id: UUID) -> Document:
        """Create a sample document for testing."""
        from app.domain import DocumentMetadata

        return Document.create(
            library_id=library_id,
            title="Test Document",
            content="This is test content",
            metadata=DocumentMetadata(author="Test Author"),
        )

    def test_document_create_success(self, library_id: UUID):
        """Test successful document creation."""
        from app.domain import DocumentMetadata

        document = Document.create(
            library_id=library_id,
            title="Test Document",
            content="Test content",
            metadata=DocumentMetadata(tags=["test"]),
        )

        assert isinstance(document.id, UUID)
        assert document.library_id == library_id
        assert document.title == "Test Document"
        assert document.content == "Test content"
        assert document.metadata == DocumentMetadata(tags=["test"])

    def test_document_create_minimal(self, library_id: UUID):
        """Test document creation with minimal required data."""
        from app.domain import DocumentMetadata

        document = Document.create(
            library_id=library_id,
            title="Minimal Document",
        )

        assert isinstance(document.id, UUID)
        assert document.library_id == library_id
        assert document.title == "Minimal Document"
        assert document.content == ""
        assert document.metadata == DocumentMetadata()

    def test_document_requires_library_id(self):
        """Test that document requires a valid library_id."""
        library_id = uuid.uuid4()
        document = Document.create(
            library_id=library_id,
            title="Test Document",
        )
        assert document.library_id == library_id

    def test_document_title_validation_empty(self, library_id: UUID):
        """Test that empty title raises ValueError."""
        with pytest.raises(ValueError, match="Document title cannot be empty"):
            Document.create(
                library_id=library_id,
                title="",
            )

    def test_document_title_validation_whitespace(self, library_id: UUID):
        """Test that whitespace-only title raises ValueError."""
        with pytest.raises(ValueError, match="Document title cannot be empty"):
            Document.create(
                library_id=library_id,
                title="   ",
            )

    def test_document_title_validation_too_long(self, library_id: UUID):
        """Test that too long title raises ValueError."""
        long_title = "x" * 256
        with pytest.raises(
            ValueError, match="Document title cannot exceed 255 characters"
        ):
            Document.create(
                library_id=library_id,
                title=long_title,
            )

    def test_document_title_strips_whitespace(self, library_id: UUID):
        """Test that document title is stripped of whitespace."""
        document = Document.create(
            library_id=library_id,
            title="  Test Document  ",
        )
        assert document.title == "Test Document"

    def test_document_metadata_default_empty_dict(self, library_id: UUID):
        """Test that metadata defaults to empty DocumentMetadata."""
        from app.domain import DocumentMetadata

        document = Document.create(
            library_id=library_id,
            title="Test Document",
        )
        assert document.metadata == DocumentMetadata()

    def test_document_update_success(self, sample_document: Document):
        """Test successful document update."""
        from app.domain import DocumentMetadata

        updated_document = sample_document.update(
            title="Updated Title",
            content="Updated content",
            metadata=DocumentMetadata(tags=["updated"]),
        )

        # Original document unchanged
        assert sample_document.title == "Test Document"
        assert sample_document.content == "This is test content"
        assert sample_document.metadata == DocumentMetadata(author="Test Author")

        # New document has updated values
        assert updated_document.id == sample_document.id  # Same ID
        assert (
            updated_document.library_id == sample_document.library_id
        )  # Same library_id
        assert updated_document.title == "Updated Title"
        assert updated_document.content == "Updated content"
        assert updated_document.metadata == DocumentMetadata(tags=["updated"])

    def test_document_update_partial(self, sample_document: Document):
        """Test partial document update."""
        updated_document = sample_document.update(title="New Title Only")

        # Only title should change
        assert updated_document.title == "New Title Only"
        assert updated_document.content == sample_document.content
        assert updated_document.metadata == sample_document.metadata

    def test_document_update_returns_new_instance(self, sample_document: Document):
        """Test that update returns a new instance and keeps old unchanged."""
        updated_document = sample_document.update(title="New Title")

        # Different objects
        assert updated_document is not sample_document
        assert id(updated_document) != id(sample_document)

        # Original unchanged
        assert sample_document.title == "Test Document"

    def test_document_immutable(self, sample_document: Document):
        """Test that document is immutable (frozen dataclass)."""
        with pytest.raises(AttributeError):
            sample_document.title = "Cannot change"

    def test_document_validation_preserves_id_and_library_id(
        self, sample_document: Document
    ):
        """Test that update preserves ID and library_id."""
        updated_document = sample_document.update(title="New Title")

        assert updated_document.id == sample_document.id
        assert updated_document.library_id == sample_document.library_id

    def test_document_update_returns_new_instance_and_keeps_old_unchanged(self):
        """Test that Document.update() returns new instance and keeps old unchanged."""
        library_id = uuid.uuid4()
        from app.domain import DocumentMetadata

        original_document = Document.create(
            library_id=library_id,
            title="Original Document",
            content="Original content",
            metadata=DocumentMetadata(tags=["original"]),
        )

        updated_document = original_document.update(
            title="Updated Document",
            content="Updated content",
            metadata=DocumentMetadata(tags=["updated"]),
        )

        # Different instances
        assert updated_document is not original_document
        assert id(updated_document) != id(original_document)

        # Original unchanged
        assert original_document.title == "Original Document"
        assert original_document.content == "Original content"
        assert original_document.metadata == DocumentMetadata(tags=["original"])

        # Updated has new values
        assert updated_document.title == "Updated Document"
        assert updated_document.content == "Updated content"
        assert updated_document.metadata == DocumentMetadata(tags=["updated"])

        # Same ID and library_id
        assert updated_document.id == original_document.id
        assert updated_document.library_id == original_document.library_id


class TestDocumentIntegrationWithLibrary:
    """Test suite for Document integration with Library."""

    def test_document_can_be_created_with_library_reference(self):
        """Test that Document can reference an existing Library."""
        # Create a library first
        library = Library.create(
            name="Test Library",
            description="A library for testing documents",
        )

        # Create a document that references the library
        document = Document.create(
            library_id=library.id,
            title="Document in Library",
            content="This document belongs to the test library",
        )

        assert document.library_id == library.id
        assert isinstance(document.id, UUID)
        assert document.id != library.id  # Different entities have different IDs

    def test_document_preserves_library_relationship_on_update(self):
        """Test that Document maintains library relationship after updates."""
        library = Library.create(name="Persistent Library")
        original_document = Document.create(
            library_id=library.id,
            title="Original Title",
        )

        updated_document = original_document.update(
            title="Updated Title",
            content="New content added",
        )

        # Library relationship preserved
        assert updated_document.library_id == library.id
        assert updated_document.library_id == original_document.library_id
