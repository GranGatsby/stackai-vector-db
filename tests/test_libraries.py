"""End-to-end tests for the libraries API."""

import uuid
from uuid import UUID

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from app.api.v1.deps import get_library_repository
from app.main import app


@pytest.fixture
def client():
    """Create a fresh test client for each test."""
    test_client = TestClient(app)
    # Clear the repository before each test
    repo = get_library_repository()
    repo.clear()
    return test_client


class TestLibrariesAPI:
    """Test suite for the libraries API endpoints."""

    def test_list_empty_libraries(self, client: TestClient):
        """Test listing libraries when none exist."""
        response = client.get("/api/v1/libraries/")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data == {"libraries": [], "total": 0, "limit": None, "offset": 0}

    def test_create_library_success(self, client: TestClient):
        """Test successful library creation."""
        library_data = {
            "name": "Test Library",
            "description": "A test library",
            "metadata": {"author": "Test Author"},
        }

        response = client.post("/api/v1/libraries/", json=library_data)

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()

        # Validate response structure
        assert "id" in data
        assert UUID(data["id"])  # Validate it's a valid UUID
        assert data["name"] == library_data["name"]
        assert data["description"] == library_data["description"]
        assert data["metadata"]["author"] == library_data["metadata"]["author"]

    def test_create_library_minimal(self, client: TestClient):
        """Test creating a library with minimal required data."""
        library_data = {"name": "Minimal Library"}

        response = client.post("/api/v1/libraries/", json=library_data)

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()

        assert "id" in data
        assert data["name"] == "Minimal Library"
        assert data["description"] == ""
        # Metadata is now a full schema with all fields (most None)
        assert isinstance(data["metadata"], dict)
        assert data["metadata"]["author"] is None

    def test_create_library_duplicate_name(self, client: TestClient):
        """Test creating a library with a duplicate name fails."""
        library_data = {"name": "Duplicate Library"}

        # Create first library
        response1 = client.post("/api/v1/libraries/", json=library_data)
        assert response1.status_code == status.HTTP_201_CREATED

        # Try to create second library with same name
        response2 = client.post("/api/v1/libraries/", json=library_data)
        assert response2.status_code == status.HTTP_409_CONFLICT

        data = response2.json()
        assert data["error"]["code"] == "CONFLICT"
        assert "already exists" in data["error"]["message"].lower()

    def test_create_library_invalid_name(self, client: TestClient):
        """Test creating a library with invalid name fails."""
        test_cases = [
            {"name": ""},  # Empty name
            {"name": "   "},  # Whitespace only
            {"name": "x" * 256},  # Too long
        ]

        for library_data in test_cases:
            response = client.post("/api/v1/libraries/", json=library_data)
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

    def test_get_library_success(self, client: TestClient):
        """Test successful library retrieval."""
        # First create a library
        library_data = {
            "name": "Get Test Library",
            "description": "For testing get endpoint",
            "metadata": {"test": True},
        }
        create_response = client.post("/api/v1/libraries/", json=library_data)
        assert create_response.status_code == status.HTTP_201_CREATED
        library_id = create_response.json()["id"]

        # Then retrieve it
        response = client.get(f"/api/v1/libraries/{library_id}")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["id"] == library_id
        assert data["name"] == library_data["name"]
        assert data["description"] == library_data["description"]
        # Check specific metadata field that was set
        assert data["metadata"]["test"] == library_data["metadata"]["test"]

    def test_get_library_not_found(self, client: TestClient):
        """Test retrieving a non-existent library fails."""
        non_existent_id = str(uuid.uuid4())
        response = client.get(f"/api/v1/libraries/{non_existent_id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert data["error"]["code"] == "NOT_FOUND"
        assert non_existent_id in data["error"]["message"]

    def test_get_library_invalid_uuid(self, client: TestClient):
        """Test retrieving a library with invalid UUID format fails."""
        response = client.get("/api/v1/libraries/invalid-uuid")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

    def test_update_library_success(self, client: TestClient):
        """Test successful library update."""
        # Create a library
        create_data = {
            "name": "Original Library",
            "description": "Original description",
        }
        create_response = client.post("/api/v1/libraries/", json=create_data)
        library_id = create_response.json()["id"]

        # Update it
        update_data = {
            "name": "Updated Library",
            "description": "Updated description",
            "metadata": {"updated": True},
        }
        response = client.patch(f"/api/v1/libraries/{library_id}", json=update_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["id"] == library_id
        assert data["name"] == update_data["name"]
        assert data["description"] == update_data["description"]
        # Check specific metadata field that was set
        assert data["metadata"]["updated"] == update_data["metadata"]["updated"]

    def test_update_library_partial(self, client: TestClient):
        """Test partial library update."""
        # Create a library
        create_data = {
            "name": "Partial Update Library",
            "description": "Original description",
            "metadata": {"original": True},
        }
        create_response = client.post("/api/v1/libraries/", json=create_data)
        library_id = create_response.json()["id"]

        # Update only the name
        update_data = {"name": "Partially Updated Library"}
        response = client.patch(f"/api/v1/libraries/{library_id}", json=update_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["name"] == update_data["name"]
        assert data["description"] == create_data["description"]  # Unchanged
        # Check specific metadata field that was set originally
        assert (
            data["metadata"]["original"] == create_data["metadata"]["original"]
        )  # Unchanged

    def test_update_library_not_found(self, client: TestClient):
        """Test updating a non-existent library fails."""
        non_existent_id = str(uuid.uuid4())
        update_data = {"name": "Updated Name"}

        response = client.patch(
            f"/api/v1/libraries/{non_existent_id}", json=update_data
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert data["error"]["code"] == "NOT_FOUND"

    def test_update_library_duplicate_name(self, client: TestClient):
        """Test updating library to an existing name fails."""
        # Create two libraries
        lib1_data = {"name": "Library One"}
        lib2_data = {"name": "Library Two"}

        lib1_response = client.post("/api/v1/libraries/", json=lib1_data)
        lib2_response = client.post("/api/v1/libraries/", json=lib2_data)

        lib2_id = lib2_response.json()["id"]

        # Try to update lib2 to have the same name as lib1
        update_data = {"name": "Library One"}
        response = client.patch(f"/api/v1/libraries/{lib2_id}", json=update_data)

        assert response.status_code == status.HTTP_409_CONFLICT
        data = response.json()
        assert data["error"]["code"] == "CONFLICT"

    def test_update_library_same_name(self, client: TestClient):
        """Test updating library with the same name should succeed."""
        # Create a library
        create_data = {"name": "Same Name Library", "description": "Original"}
        create_response = client.post("/api/v1/libraries/", json=create_data)
        library_id = create_response.json()["id"]

        # Update with same name but different description
        update_data = {"name": "Same Name Library", "description": "Updated"}
        response = client.patch(f"/api/v1/libraries/{library_id}", json=update_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["name"] == "Same Name Library"
        assert data["description"] == "Updated"

    def test_create_library_case_insensitive_names(self, client: TestClient):
        """Test that library names are case-insensitive."""
        # Create library with uppercase name
        lib1_data = {"name": "Alpha Library"}
        response1 = client.post("/api/v1/libraries/", json=lib1_data)
        assert response1.status_code == status.HTTP_201_CREATED

        # Try to create library with lowercase name - should fail
        lib2_data = {"name": "alpha library"}
        response2 = client.post("/api/v1/libraries/", json=lib2_data)
        assert response2.status_code == status.HTTP_409_CONFLICT

        # Try mixed case - should also fail
        lib3_data = {"name": "ALPHA LIBRARY"}
        response3 = client.post("/api/v1/libraries/", json=lib3_data)
        assert response3.status_code == status.HTTP_409_CONFLICT

    def test_list_libraries_case_insensitive_sorting(self, client: TestClient):
        """Test that libraries are sorted case-insensitively."""
        # Create libraries with mixed case names
        libraries_data = [
            {"name": "zebra Library"},
            {"name": "Alpha Library"},
            {"name": "beta Library"},
            {"name": "Charlie Library"},
        ]

        for lib_data in libraries_data:
            response = client.post("/api/v1/libraries/", json=lib_data)
            assert response.status_code == status.HTTP_201_CREATED

        # List libraries and check sorting
        response = client.get("/api/v1/libraries/")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        library_names = [lib["name"] for lib in data["libraries"]]

        # Should be sorted case-insensitively
        expected_order = [
            "Alpha Library",
            "beta Library",
            "Charlie Library",
            "zebra Library",
        ]
        assert library_names == expected_order

    def test_delete_library_success(self, client: TestClient):
        """Test successful library deletion."""
        # Create a library
        create_data = {"name": "Library to Delete"}
        create_response = client.post("/api/v1/libraries/", json=create_data)
        library_id = create_response.json()["id"]

        # Delete it
        response = client.delete(f"/api/v1/libraries/{library_id}")
        assert response.status_code == status.HTTP_204_NO_CONTENT

        # Verify it's gone
        get_response = client.get(f"/api/v1/libraries/{library_id}")
        assert get_response.status_code == status.HTTP_404_NOT_FOUND

    def test_delete_library_not_found(self, client: TestClient):
        """Test deleting a non-existent library fails."""
        non_existent_id = str(uuid.uuid4())
        response = client.delete(f"/api/v1/libraries/{non_existent_id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_list_libraries_with_data(self, client: TestClient):
        """Test listing libraries when some exist."""
        # Create multiple libraries
        libraries_data = [
            {"name": "Alpha Library", "description": "First library"},
            {"name": "Beta Library", "description": "Second library"},
            {"name": "Gamma Library", "description": "Third library"},
        ]

        created_ids = []
        for lib_data in libraries_data:
            response = client.post("/api/v1/libraries/", json=lib_data)
            created_ids.append(response.json()["id"])

        # List all libraries
        response = client.get("/api/v1/libraries/")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["total"] == 3
        assert len(data["libraries"]) == 3

        # Verify libraries are sorted by name
        library_names = [lib["name"] for lib in data["libraries"]]
        assert library_names == sorted(library_names)

        # Verify all created libraries are present
        response_ids = {lib["id"] for lib in data["libraries"]}
        assert response_ids == set(created_ids)

    def test_list_libraries_pagination(self, client: TestClient):
        """Test library listing with pagination."""
        # Create 5 libraries
        libraries_data = [{"name": f"Library {i:02d}"} for i in range(1, 6)]

        for lib_data in libraries_data:
            response = client.post("/api/v1/libraries/", json=lib_data)
            assert response.status_code == status.HTTP_201_CREATED

        # Test pagination with limit=2
        response = client.get("/api/v1/libraries/?limit=2")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert len(data["libraries"]) == 2
        assert data["total"] == 5
        assert data["limit"] == 2
        assert data["offset"] == 0

        # Test pagination with limit=2, offset=2
        response = client.get("/api/v1/libraries/?limit=2&offset=2")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert len(data["libraries"]) == 2
        assert data["total"] == 5
        assert data["limit"] == 2
        assert data["offset"] == 2

        # Test pagination with offset beyond available items
        response = client.get("/api/v1/libraries/?limit=2&offset=10")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert len(data["libraries"]) == 0
        assert data["total"] == 5
        assert data["limit"] == 2
        assert data["offset"] == 10

    def test_list_libraries_pagination_validation(self, client: TestClient):
        """Test pagination parameter validation."""
        # Test invalid limit (too small)
        response = client.get("/api/v1/libraries/?limit=0")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

        # Test invalid limit (too large)
        response = client.get("/api/v1/libraries/?limit=101")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

        # Test invalid offset (negative)
        response = client.get("/api/v1/libraries/?offset=-1")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

    def test_crud_workflow(self, client: TestClient):
        """Test complete CRUD workflow."""
        # CREATE
        create_data = {
            "name": "CRUD Workflow Library",
            "description": "Testing complete workflow",
            "metadata": {"workflow": "test"},
        }
        create_response = client.post("/api/v1/libraries/", json=create_data)
        assert create_response.status_code == status.HTTP_201_CREATED
        library_id = create_response.json()["id"]

        # READ (get single)
        get_response = client.get(f"/api/v1/libraries/{library_id}")
        assert get_response.status_code == status.HTTP_200_OK
        assert get_response.json()["name"] == create_data["name"]

        # READ (list of libraries)
        list_response = client.get("/api/v1/libraries/")
        assert list_response.status_code == status.HTTP_200_OK
        assert list_response.json()["total"] == 1

        # UPDATE
        update_data = {"name": "Updated CRUD Library"}
        update_response = client.patch(
            f"/api/v1/libraries/{library_id}", json=update_data
        )
        assert update_response.status_code == status.HTTP_200_OK
        assert update_response.json()["name"] == update_data["name"]

        # DELETE
        delete_response = client.delete(f"/api/v1/libraries/{library_id}")
        assert delete_response.status_code == status.HTTP_204_NO_CONTENT

        # Verify deletion
        final_list_response = client.get("/api/v1/libraries/")
        assert final_list_response.json()["total"] == 0
