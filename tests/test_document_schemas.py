"""Unit tests for document schemas."""

import uuid

import pytest
from pydantic import ValidationError

from app.schemas import DocumentCreate, DocumentRead, DocumentUpdate


class TestDocumentSchemas:
    """Test suite for document schemas."""

    def test_document_create_valid(self):
        """Test valid document creation schema."""
        library_id = uuid.uuid4()
        data = {
            "library_id": library_id,
            "title": "Test Document",
            "content": "Test content",
            "metadata": {"author": "Test Author"},
        }

        schema = DocumentCreate(**data)
        assert schema.library_id == library_id
        assert schema.title == "Test Document"
        assert schema.content == "Test content"
        assert schema.metadata["author"] == "Test Author"

    def test_document_create_minimal(self):
        """Test document creation with minimal required fields."""
        library_id = uuid.uuid4()
        data = {
            "library_id": library_id,
            "title": "Minimal Document",
        }

        schema = DocumentCreate(**data)
        assert schema.library_id == library_id
        assert schema.title == "Minimal Document"
        assert schema.content == ""  # Default
        assert schema.metadata == {}  # Default

    def test_document_create_invalid_title(self):
        """Test document creation with invalid title."""
        library_id = uuid.uuid4()

        # Empty title
        with pytest.raises(ValidationError, match="at least 1 character"):
            DocumentCreate(library_id=library_id, title="")

        # Whitespace only title
        with pytest.raises(ValidationError, match="empty or whitespace"):
            DocumentCreate(library_id=library_id, title="   ")

        # Title too long
        with pytest.raises(ValidationError, match="at most 255 characters"):
            DocumentCreate(library_id=library_id, title="a" * 256)

    def test_document_update_partial(self):
        """Test partial document update schema."""
        data = {
            "title": "Updated Title",
            "content": "Updated content",
        }

        schema = DocumentUpdate(**data)
        assert schema.title == "Updated Title"
        assert schema.content == "Updated content"
        assert schema.metadata is None  # Not provided

    def test_document_update_empty_optional(self):
        """Test document update with no fields provided."""
        schema = DocumentUpdate()
        assert schema.title is None
        assert schema.content is None
        assert schema.metadata is None

    def test_document_update_invalid_title(self):
        """Test document update with invalid title."""
        # Empty title
        with pytest.raises(ValidationError, match="at least 1 character"):
            DocumentUpdate(title="")

        # Whitespace only title
        with pytest.raises(ValidationError, match="empty or whitespace"):
            DocumentUpdate(title="   ")

    def test_document_read_from_domain(self):
        """Test DocumentRead creation from domain entity."""
        from app.domain import Document

        library_id = uuid.uuid4()
        from app.domain import DocumentMetadata

        document = Document.create(
            library_id=library_id,
            title="Test Document",
            content="Test content",
            metadata=DocumentMetadata(author="Test Author"),
        )

        schema = DocumentRead.from_domain(document)
        assert schema.id == document.id
        assert schema.library_id == document.library_id
        assert schema.title == document.title
        assert schema.content == document.content
        # metadata is converted to dict in schema response
        assert "author" in schema.metadata
        assert schema.metadata["author"] == "Test Author"
