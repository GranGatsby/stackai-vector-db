"""Unit tests for document schemas."""

import uuid

import pytest
from pydantic import ValidationError

from app.schemas.document import (
    DocumentCreateInLibrary,
    DocumentMetadataSchema,
    DocumentRead,
    DocumentUpdate,
)


class TestDocumentSchemas:
    """Test suite for document schemas."""

    def test_document_create_in_library_valid(self):
        """Test valid document creation in library schema."""
        data = {
            "title": "Test Document",
            "content": "Test content",
            "metadata": {"author": "Test Author"},
        }

        schema = DocumentCreateInLibrary(**data)
        assert schema.title == "Test Document"
        assert schema.content == "Test content"
        assert schema.metadata.author == "Test Author"

    def test_document_create_in_library_minimal(self):
        """Test document creation in library with minimal required fields."""
        data = {
            "title": "Minimal Document",
        }

        schema = DocumentCreateInLibrary(**data)
        assert schema.title == "Minimal Document"
        assert schema.content == ""  # Default
        assert isinstance(
            schema.metadata, DocumentMetadataSchema
        )

    def test_document_create_in_library_invalid_title(self):
        """Test document creation in library with invalid title."""
        # Empty title
        with pytest.raises(ValidationError, match="at least 1 character"):
            DocumentCreateInLibrary(title="")

        # Whitespace only title
        with pytest.raises(ValidationError, match="empty or whitespace"):
            DocumentCreateInLibrary(title="   ")

        # Title too long
        with pytest.raises(ValidationError, match="at most 255 characters"):
            DocumentCreateInLibrary(title="a" * 256)

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
        # metadata is now a DocumentMetadataSchema object
        assert isinstance(schema.metadata, DocumentMetadataSchema)
        assert schema.metadata.author == "Test Author"
