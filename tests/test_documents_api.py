"""Integration tests for Documents API endpoints."""

import uuid

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from app.main import app


class TestDocumentsAPI:
    """Test suite for Documents API endpoints."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        return TestClient(app)

    @pytest.fixture
    def sample_library(self, client: TestClient) -> dict:
        """Create a sample library for testing."""
        # Use unique name to avoid conflicts across tests
        library_name = f"Test Library Docs {uuid.uuid4().hex[:8]}"
        response = client.post(
            "/api/v1/libraries/",
            json={"name": library_name, "description": "For document testing"},
        )
        assert response.status_code == status.HTTP_201_CREATED
        return response.json()

    def test_create_document_success(self, client: TestClient, sample_library: dict):
        """Test successful document creation in a library."""
        document_data = {
            "title": "Test Document",
            "content": "This is test content for the document",
            "metadata": {"author": "Test Author", "category": "test"},
        }

        response = client.post(
            f"/api/v1/libraries/{sample_library['id']}/documents",
            json=document_data,
        )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["title"] == "Test Document"
        assert data["content"] == "This is test content for the document"
        assert data["library_id"] == sample_library["id"]
        assert "id" in data
        assert data["metadata"]["author"] == "Test Author"
        assert data["metadata"]["category"] == "test"

    def test_get_document_success(self, client: TestClient, sample_library: dict):
        """Test successful document retrieval by ID."""
        # Create document first
        create_response = client.post(
            f"/api/v1/libraries/{sample_library['id']}/documents",
            json={"title": "Get Test Document", "content": "Content to retrieve"},
        )
        created_doc = create_response.json()

        # Get document
        response = client.get(f"/api/v1/documents/{created_doc['id']}")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == created_doc["id"]
        assert data["title"] == "Get Test Document"
        assert data["content"] == "Content to retrieve"
        assert data["library_id"] == sample_library["id"]

    def test_get_document_not_found(self, client: TestClient):
        """Test document retrieval with non-existent ID."""
        non_existent_id = str(uuid.uuid4())
        response = client.get(f"/api/v1/documents/{non_existent_id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert "not found" in data["error"]["message"].lower()

    def test_update_document_success(self, client: TestClient, sample_library: dict):
        """Test successful document update."""
        # Create document first
        create_response = client.post(
            f"/api/v1/libraries/{sample_library['id']}/documents",
            json={"title": "Original Title", "content": "Original content"},
        )
        created_doc = create_response.json()

        # Update document
        update_data = {
            "title": "Updated Title",
            "content": "Updated content",
            "metadata": {"author": "Updated Author"},
        }
        response = client.patch(
            f"/api/v1/documents/{created_doc['id']}", json=update_data
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == created_doc["id"]
        assert data["title"] == "Updated Title"
        assert data["content"] == "Updated content"
        assert data["metadata"]["author"] == "Updated Author"
        assert data["library_id"] == sample_library["id"]

    def test_delete_document_success(self, client: TestClient, sample_library: dict):
        """Test successful document deletion."""
        # Create document first
        create_response = client.post(
            f"/api/v1/libraries/{sample_library['id']}/documents",
            json={"title": "To Delete", "content": "Will be deleted"},
        )
        created_doc = create_response.json()

        # Delete document
        response = client.delete(f"/api/v1/documents/{created_doc['id']}")

        assert response.status_code == status.HTTP_204_NO_CONTENT

        # Verify document is gone
        get_response = client.get(f"/api/v1/documents/{created_doc['id']}")
        assert get_response.status_code == status.HTTP_404_NOT_FOUND

    def test_list_documents_by_library(self, client: TestClient, sample_library: dict):
        """Test listing documents by library with pagination."""
        # Create multiple documents
        for i in range(3):
            client.post(
                f"/api/v1/libraries/{sample_library['id']}/documents",
                json={"title": f"Document {i:02d}", "content": f"Content {i}"},
            )

        # List all documents
        response = client.get(f"/api/v1/libraries/{sample_library['id']}/documents")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] == 3
        assert len(data["documents"]) == 3
        assert data["limit"] is None
        assert data["offset"] == 0

        # Test pagination
        paginated_response = client.get(
            f"/api/v1/libraries/{sample_library['id']}/documents?limit=2&offset=1"
        )
        assert paginated_response.status_code == status.HTTP_200_OK
        paginated_data = paginated_response.json()
        assert paginated_data["total"] == 3
        assert len(paginated_data["documents"]) == 2
        assert paginated_data["limit"] == 2
        assert paginated_data["offset"] == 1
