"""Unit tests for the InMemoryDocumentRepository."""

import uuid

import pytest

from app.domain import Document, DocumentNotFoundError
from app.repositories.in_memory import InMemoryDocumentRepository


class TestInMemoryDocumentRepository:
    """Test suite for InMemoryDocumentRepository."""

    @pytest.fixture
    def repository(self):
        """Create a fresh repository for each test."""
        return InMemoryDocumentRepository()

    @pytest.fixture
    def library_id(self):
        """Create a sample library ID for testing."""
        return uuid.uuid4()

    @pytest.fixture
    def sample_document(self, library_id):
        """Create a sample document for testing."""
        return Document.create(
            library_id=library_id,
            title="Test Document",
            content="This is test content",
            metadata={"author": "Test Author"},
        )

    def test_document_repo_crud_roundtrip(
        self, repository: InMemoryDocumentRepository, sample_document: Document
    ):
        """Test complete CRUD roundtrip for documents."""
        # CREATE
        created = repository.create(sample_document)
        assert created == sample_document
        assert repository.exists(sample_document.id)
        assert repository.count_by_library(sample_document.library_id) == 1

        # READ
        retrieved = repository.get_by_id(sample_document.id)
        assert retrieved == sample_document

        # UPDATE
        updated_document = sample_document.update(
            title="Updated Title", content="Updated content"
        )
        result = repository.update(updated_document)
        assert result == updated_document
        assert repository.get_by_id(sample_document.id).title == "Updated Title"

        # DELETE
        deleted = repository.delete(sample_document.id)
        assert deleted is True
        assert not repository.exists(sample_document.id)
        assert repository.get_by_id(sample_document.id) is None
        assert repository.count_by_library(sample_document.library_id) == 0

    def test_list_documents_by_library_with_pagination(
        self, repository: InMemoryDocumentRepository, library_id
    ):
        """Test listing documents by library with pagination."""
        # Create multiple documents in the same library
        documents = []
        for i in range(5):
            doc = Document.create(
                library_id=library_id,
                title=f"Document {i:02d}",
                content=f"Content {i}",
            )
            repository.create(doc)
            documents.append(doc)

        # Test listing all documents
        all_docs = repository.list_by_library(library_id)
        assert len(all_docs) == 5
        # Should be sorted by title
        assert all_docs[0].title == "Document 00"
        assert all_docs[-1].title == "Document 04"

        # Test pagination with limit
        limited = repository.list_by_library(library_id, limit=2)
        assert len(limited) == 2
        assert limited[0].title == "Document 00"
        assert limited[1].title == "Document 01"

        # Test pagination with offset
        offset_docs = repository.list_by_library(library_id, offset=2)
        assert len(offset_docs) == 3
        assert offset_docs[0].title == "Document 02"

        # Test pagination with both limit and offset
        page_docs = repository.list_by_library(library_id, limit=2, offset=1)
        assert len(page_docs) == 2
        assert page_docs[0].title == "Document 01"
        assert page_docs[1].title == "Document 02"

        # Test count
        assert repository.count_by_library(library_id) == 5

    def test_delete_document_updates_secondary_indexes(
        self, repository: InMemoryDocumentRepository
    ):
        """Test that deleting documents updates secondary indexes properly."""
        lib1_id = uuid.uuid4()
        lib2_id = uuid.uuid4()

        # Create documents in different libraries
        doc1 = Document.create(library_id=lib1_id, title="Doc 1")
        doc2 = Document.create(library_id=lib1_id, title="Doc 2")
        doc3 = Document.create(library_id=lib2_id, title="Doc 3")

        repository.create(doc1)
        repository.create(doc2)
        repository.create(doc3)

        # Verify initial state
        assert repository.count_by_library(lib1_id) == 2
        assert repository.count_by_library(lib2_id) == 1

        # Delete one document from lib1
        repository.delete(doc1.id)

        # Verify indexes are updated
        assert repository.count_by_library(lib1_id) == 1
        assert repository.count_by_library(lib2_id) == 1
        assert len(repository.list_by_library(lib1_id)) == 1
        assert repository.list_by_library(lib1_id)[0].id == doc2.id

        # Delete all documents from lib1
        deleted_count = repository.delete_by_library(lib1_id)
        assert deleted_count == 1
        assert repository.count_by_library(lib1_id) == 0
        assert repository.count_by_library(lib2_id) == 1

    def test_document_not_found_error(self, repository: InMemoryDocumentRepository):
        """Test that updating non-existent document raises error."""
        non_existent_doc = Document.create(
            library_id=uuid.uuid4(), title="Non-existent"
        )

        with pytest.raises(DocumentNotFoundError):
            repository.update(non_existent_doc)

    def test_document_already_exists_error(
        self, repository: InMemoryDocumentRepository, sample_document: Document
    ):
        """Test that creating duplicate document raises error."""
        repository.create(sample_document)

        with pytest.raises(ValueError, match="already exists"):
            repository.create(sample_document)

    def test_update_document_library_id_updates_indexes(
        self, repository: InMemoryDocumentRepository
    ):
        """Test that updating document library_id updates secondary indexes."""
        old_library_id = uuid.uuid4()
        new_library_id = uuid.uuid4()

        # Create document in old library
        document = Document.create(library_id=old_library_id, title="Test Doc")
        repository.create(document)

        assert repository.count_by_library(old_library_id) == 1
        assert repository.count_by_library(new_library_id) == 0

        # Update document to new library
        updated_document = Document(
            id=document.id,
            library_id=new_library_id,
            title=document.title,
            content=document.content,
            metadata=document.metadata,
        )
        repository.update(updated_document)

        # Verify indexes are updated
        assert repository.count_by_library(old_library_id) == 0
        assert repository.count_by_library(new_library_id) == 1
        assert repository.list_by_library(new_library_id)[0].id == document.id
