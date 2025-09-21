"""Unit tests for the LibraryService."""

import uuid
from unittest.mock import Mock, MagicMock
from uuid import UUID

import pytest

from app.domain import Library, LibraryNotFoundError, LibraryAlreadyExistsError
from app.services import LibraryService


class TestLibraryService:
    """Test suite for LibraryService."""

    @pytest.fixture
    def mock_repository(self):
        """Create a mock repository for testing."""
        return Mock()

    @pytest.fixture
    def service(self, mock_repository):
        """Create a service with mocked repository."""
        return LibraryService(mock_repository)

    @pytest.fixture
    def sample_library(self):
        """Create a sample library for testing."""
        return Library.create(
            name="Test Library",
            description="A test library",
            metadata={"author": "Test Author"},
        )

    def test_list_libraries(self, service: LibraryService, mock_repository):
        """Test listing libraries."""
        # Setup mock
        mock_libraries = [
            Library.create(name="Library 1"),
            Library.create(name="Library 2"),
        ]
        mock_repository.list_all.return_value = mock_libraries
        mock_repository.count_all.return_value = 2

        # Call service
        libraries, total = service.list_libraries()

        # Verify
        assert libraries == mock_libraries
        assert total == 2
        mock_repository.list_all.assert_called_once_with(limit=None, offset=0)
        mock_repository.count_all.assert_called_once()

    def test_list_libraries_with_pagination(
        self, service: LibraryService, mock_repository
    ):
        """Test listing libraries with pagination."""
        # Setup mock
        mock_libraries = [Library.create(name="Library 1")]
        mock_repository.list_all.return_value = mock_libraries
        mock_repository.count_all.return_value = 5

        # Call service with pagination
        libraries, total = service.list_libraries(limit=1, offset=2)

        # Verify
        assert libraries == mock_libraries
        assert total == 5
        mock_repository.list_all.assert_called_once_with(limit=1, offset=2)
        mock_repository.count_all.assert_called_once()

    def test_get_library_success(
        self, service: LibraryService, mock_repository, sample_library: Library
    ):
        """Test successful library retrieval."""
        # Setup mock
        mock_repository.get_by_id.return_value = sample_library

        # Call service
        result = service.get_library(sample_library.id)

        # Verify
        assert result == sample_library
        mock_repository.get_by_id.assert_called_once_with(sample_library.id)

    def test_get_library_not_found(self, service: LibraryService, mock_repository):
        """Test library retrieval when not found."""
        # Setup mock
        library_id = uuid.uuid4()
        mock_repository.get_by_id.return_value = None

        # Call service and expect exception
        with pytest.raises(LibraryNotFoundError) as exc_info:
            service.get_library(library_id)

        assert exc_info.value.library_id == str(library_id)
        mock_repository.get_by_id.assert_called_once_with(library_id)

    def test_get_library_by_name_success(
        self, service: LibraryService, mock_repository, sample_library: Library
    ):
        """Test successful library retrieval by name."""
        # Setup mock
        mock_repository.get_by_name.return_value = sample_library

        # Call service
        result = service.get_library_by_name(sample_library.name)

        # Verify
        assert result == sample_library
        mock_repository.get_by_name.assert_called_once_with(sample_library.name)

    def test_get_library_by_name_not_found(
        self, service: LibraryService, mock_repository
    ):
        """Test library retrieval by name when not found."""
        # Setup mock
        library_name = "Non-existent Library"
        mock_repository.get_by_name.return_value = None

        # Call service and expect exception
        with pytest.raises(LibraryNotFoundError) as exc_info:
            service.get_library_by_name(library_name)

        assert f"Library with name '{library_name}' not found" in str(exc_info.value)
        mock_repository.get_by_name.assert_called_once_with(library_name)

    def test_create_library_success(self, service: LibraryService, mock_repository):
        """Test successful library creation."""
        # Setup mock
        created_library = Library.create(name="New Library", description="Description")
        mock_repository.create.return_value = created_library

        # Call service
        result = service.create_library(
            name="New Library", description="Description", metadata={"test": True}
        )

        # Verify
        assert result == created_library
        mock_repository.create.assert_called_once()

        # Verify the library passed to repository has correct attributes
        call_args = mock_repository.create.call_args[0][0]
        assert call_args.name == "New Library"
        assert call_args.description == "Description"
        assert call_args.metadata == {"test": True}

    def test_create_library_minimal(self, service: LibraryService, mock_repository):
        """Test creating library with minimal data."""
        # Setup mock
        created_library = Library.create(name="Minimal Library")
        mock_repository.create.return_value = created_library

        # Call service with minimal data
        result = service.create_library(name="Minimal Library")

        # Verify
        assert result == created_library
        mock_repository.create.assert_called_once()

        # Verify defaults
        call_args = mock_repository.create.call_args[0][0]
        assert call_args.name == "Minimal Library"
        assert call_args.description == ""
        assert call_args.metadata == {}

    def test_create_library_already_exists(
        self, service: LibraryService, mock_repository
    ):
        """Test creating library when name already exists."""
        # Setup mock
        mock_repository.create.side_effect = LibraryAlreadyExistsError(
            "Duplicate Library"
        )

        # Call service and expect exception
        with pytest.raises(LibraryAlreadyExistsError):
            service.create_library(name="Duplicate Library")

        mock_repository.create.assert_called_once()

    def test_create_library_invalid_name(
        self, service: LibraryService, mock_repository
    ):
        """Test creating library with invalid name."""
        # Call service with invalid name and expect validation error
        with pytest.raises(ValueError):
            service.create_library(name="")  # Empty name should fail

        # Repository should not be called for invalid input
        mock_repository.create.assert_not_called()

    def test_update_library_success(
        self, service: LibraryService, mock_repository, sample_library: Library
    ):
        """Test successful library update."""
        # Setup mock
        mock_repository.get_by_id.return_value = sample_library
        updated_library = sample_library.update(description="Updated")
        mock_repository.update.return_value = updated_library

        # Call service
        result = service.update_library(
            library_id=sample_library.id, description="Updated"
        )

        # Verify
        assert result == updated_library
        mock_repository.get_by_id.assert_called_once_with(sample_library.id)
        mock_repository.update.assert_called_once()

    def test_update_library_not_found(self, service: LibraryService, mock_repository):
        """Test updating non-existent library."""
        # Setup mock
        library_id = uuid.uuid4()
        mock_repository.get_by_id.return_value = None

        # Call service and expect exception
        with pytest.raises(LibraryNotFoundError):
            service.update_library(library_id=library_id, name="New Name")

        mock_repository.get_by_id.assert_called_once_with(library_id)
        mock_repository.update.assert_not_called()

    def test_update_library_partial(
        self, service: LibraryService, mock_repository, sample_library: Library
    ):
        """Test partial library update."""
        # Setup mock
        mock_repository.get_by_id.return_value = sample_library
        updated_library = sample_library.update(name="New Name")
        mock_repository.update.return_value = updated_library

        # Call service with only name update
        result = service.update_library(library_id=sample_library.id, name="New Name")

        # Verify
        assert result == updated_library
        mock_repository.update.assert_called_once()

        # Verify that only name was updated (description and metadata unchanged)
        call_args = mock_repository.update.call_args[0][0]
        assert call_args.name == "New Name"
        assert call_args.description == sample_library.description
        assert call_args.metadata == sample_library.metadata

    def test_update_library_name_conflict(
        self, service: LibraryService, mock_repository, sample_library: Library
    ):
        """Test updating library to conflicting name."""
        # Setup mock
        mock_repository.get_by_id.return_value = sample_library
        mock_repository.update.side_effect = LibraryAlreadyExistsError(
            "Conflicting Name"
        )

        # Call service and expect exception
        with pytest.raises(LibraryAlreadyExistsError):
            service.update_library(
                library_id=sample_library.id, name="Conflicting Name"
            )

        mock_repository.get_by_id.assert_called_once()
        mock_repository.update.assert_called_once()

    def test_delete_library(self, service: LibraryService, mock_repository):
        """Test library deletion."""
        # Setup mock
        library_id = uuid.uuid4()
        mock_repository.delete.return_value = True

        # Call service
        result = service.delete_library(library_id)

        # Verify
        assert result is True
        mock_repository.delete.assert_called_once_with(library_id)

    def test_delete_library_not_found(self, service: LibraryService, mock_repository):
        """Test deleting non-existent library."""
        # Setup mock
        library_id = uuid.uuid4()
        mock_repository.delete.return_value = False

        # Call service
        result = service.delete_library(library_id)

        # Verify
        assert result is False
        mock_repository.delete.assert_called_once_with(library_id)

    def test_library_exists(self, service: LibraryService, mock_repository):
        """Test checking library existence."""
        # Setup mock
        library_id = uuid.uuid4()
        mock_repository.exists.return_value = True

        # Call service
        result = service.library_exists(library_id)

        # Verify
        assert result is True
        mock_repository.exists.assert_called_once_with(library_id)

    def test_library_name_exists(
        self, service: LibraryService, mock_repository, sample_library: Library
    ):
        """Test checking library name existence."""
        # Setup mock
        mock_repository.get_by_name.return_value = sample_library

        # Call service
        result = service.library_name_exists(sample_library.name)

        # Verify
        assert result is True
        mock_repository.get_by_name.assert_called_once_with(sample_library.name)

    def test_library_name_not_exists(self, service: LibraryService, mock_repository):
        """Test checking non-existent library name."""
        # Setup mock
        mock_repository.get_by_name.return_value = None

        # Call service
        result = service.library_name_exists("Non-existent")

        # Verify
        assert result is False
        mock_repository.get_by_name.assert_called_once_with("Non-existent")
