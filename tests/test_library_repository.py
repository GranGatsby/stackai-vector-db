"""Unit tests for the InMemoryLibraryRepository."""

import uuid
from uuid import UUID

import pytest

from app.domain import Library, LibraryNotFoundError, LibraryAlreadyExistsError
from app.repositories.in_memory import InMemoryLibraryRepository


class TestInMemoryLibraryRepository:
    """Test suite for InMemoryLibraryRepository."""
    
    @pytest.fixture
    def repository(self):
        """Create a fresh repository for each test."""
        return InMemoryLibraryRepository()
    
    @pytest.fixture
    def sample_library(self):
        """Create a sample library for testing."""
        return Library.create(
            name="Test Library",
            description="A test library",
            metadata={"author": "Test Author"}
        )
    
    def test_create_library_success(self, repository: InMemoryLibraryRepository, sample_library: Library):
        """Test successful library creation."""
        result = repository.create(sample_library)
        
        assert result == sample_library
        assert repository.exists(sample_library.id)
        assert repository.count_all() == 1
    
    def test_create_library_duplicate_name_fails(self, repository: InMemoryLibraryRepository):
        """Test that creating libraries with duplicate names fails."""
        lib1 = Library.create(name="Duplicate Library")
        lib2 = Library.create(name="Duplicate Library")
        
        repository.create(lib1)
        
        with pytest.raises(LibraryAlreadyExistsError) as exc_info:
            repository.create(lib2)
        
        assert exc_info.value.name == "Duplicate Library"
        assert repository.count_all() == 1
    
    def test_create_library_case_insensitive_names(self, repository: InMemoryLibraryRepository):
        """Test that library names are case-insensitive."""
        lib1 = Library.create(name="Alpha Library")
        lib2 = Library.create(name="alpha library")
        lib3 = Library.create(name="ALPHA LIBRARY")
        
        repository.create(lib1)
        
        # Both should fail due to case-insensitive comparison
        with pytest.raises(LibraryAlreadyExistsError):
            repository.create(lib2)
        
        with pytest.raises(LibraryAlreadyExistsError):
            repository.create(lib3)
    
    def test_get_by_id_success(self, repository: InMemoryLibraryRepository, sample_library: Library):
        """Test successful retrieval by ID."""
        repository.create(sample_library)
        
        result = repository.get_by_id(sample_library.id)
        
        assert result == sample_library
    
    def test_get_by_id_not_found(self, repository: InMemoryLibraryRepository):
        """Test retrieval of non-existent library by ID."""
        non_existent_id = uuid.uuid4()
        
        result = repository.get_by_id(non_existent_id)
        
        assert result is None
    
    def test_get_by_name_success(self, repository: InMemoryLibraryRepository, sample_library: Library):
        """Test successful retrieval by name."""
        repository.create(sample_library)
        
        result = repository.get_by_name(sample_library.name)
        
        assert result == sample_library
    
    def test_get_by_name_case_insensitive(self, repository: InMemoryLibraryRepository):
        """Test that name lookup is case-insensitive."""
        library = Library.create(name="Alpha Library")
        repository.create(library)
        
        # All variations should find the library
        assert repository.get_by_name("Alpha Library") == library
        assert repository.get_by_name("alpha library") == library
        assert repository.get_by_name("ALPHA LIBRARY") == library
        assert repository.get_by_name("AlPhA lIbRaRy") == library
    
    def test_get_by_name_not_found(self, repository: InMemoryLibraryRepository):
        """Test retrieval of non-existent library by name."""
        result = repository.get_by_name("Non-existent Library")
        
        assert result is None
    
    def test_update_library_success(self, repository: InMemoryLibraryRepository, sample_library: Library):
        """Test successful library update."""
        repository.create(sample_library)
        
        updated_library = sample_library.update(
            description="Updated description",
            metadata={"updated": True}
        )
        
        result = repository.update(updated_library)
        
        assert result == updated_library
        assert repository.get_by_id(sample_library.id) == updated_library
    
    def test_update_library_not_found(self, repository: InMemoryLibraryRepository):
        """Test updating non-existent library fails."""
        non_existent_library = Library.create(name="Non-existent")
        
        with pytest.raises(LibraryNotFoundError):
            repository.update(non_existent_library)
    
    def test_update_library_same_name_succeeds(self, repository: InMemoryLibraryRepository):
        """Test updating library with same name succeeds."""
        library = Library.create(name="Same Name", description="Original")
        repository.create(library)
        
        updated_library = library.update(description="Updated")
        result = repository.update(updated_library)
        
        assert result.description == "Updated"
        assert result.name == "Same Name"
    
    def test_update_library_name_conflict(self, repository: InMemoryLibraryRepository):
        """Test updating library to existing name fails."""
        lib1 = Library.create(name="Library One")
        lib2 = Library.create(name="Library Two")
        
        repository.create(lib1)
        repository.create(lib2)
        
        # Try to update lib2 to have the same name as lib1
        updated_lib2 = lib2.update(name="Library One")
        
        with pytest.raises(LibraryAlreadyExistsError):
            repository.update(updated_lib2)
    
    def test_update_library_case_insensitive_conflict(self, repository: InMemoryLibraryRepository):
        """Test updating library to existing name with different case fails."""
        lib1 = Library.create(name="Alpha Library")
        lib2 = Library.create(name="Beta Library")
        
        repository.create(lib1)
        repository.create(lib2)
        
        # Try to update lib2 to have similar name with different case
        updated_lib2 = lib2.update(name="alpha library")
        
        with pytest.raises(LibraryAlreadyExistsError):
            repository.update(updated_lib2)
    
    def test_delete_library_success(self, repository: InMemoryLibraryRepository, sample_library: Library):
        """Test successful library deletion."""
        repository.create(sample_library)
        
        result = repository.delete(sample_library.id)
        
        assert result is True
        assert not repository.exists(sample_library.id)
        assert repository.count_all() == 0
    
    def test_delete_library_not_found(self, repository: InMemoryLibraryRepository):
        """Test deleting non-existent library."""
        non_existent_id = uuid.uuid4()
        
        result = repository.delete(non_existent_id)
        
        assert result is False
    
    def test_list_all_empty(self, repository: InMemoryLibraryRepository):
        """Test listing when repository is empty."""
        result = repository.list_all()
        
        assert result == []
        assert repository.count_all() == 0
    
    def test_list_all_with_libraries(self, repository: InMemoryLibraryRepository):
        """Test listing libraries."""
        libraries = [
            Library.create(name="Zebra Library"),
            Library.create(name="Alpha Library"),
            Library.create(name="Beta Library")
        ]
        
        for library in libraries:
            repository.create(library)
        
        result = repository.list_all()
        
        assert len(result) == 3
        # Should be sorted case-insensitively by name
        names = [lib.name for lib in result]
        assert names == ["Alpha Library", "Beta Library", "Zebra Library"]
    
    def test_list_all_with_pagination(self, repository: InMemoryLibraryRepository):
        """Test listing libraries with pagination."""
        # Create 5 libraries
        for i in range(1, 6):
            library = Library.create(name=f"Library {i:02d}")
            repository.create(library)
        
        # Test limit only
        result = repository.list_all(limit=2)
        assert len(result) == 2
        
        # Test offset only
        result = repository.list_all(offset=2)
        assert len(result) == 3
        
        # Test limit and offset
        result = repository.list_all(limit=2, offset=2)
        assert len(result) == 2
        
        # Test offset beyond available
        result = repository.list_all(offset=10)
        assert len(result) == 0
    
    def test_exists(self, repository: InMemoryLibraryRepository, sample_library: Library):
        """Test library existence check."""
        assert not repository.exists(sample_library.id)
        
        repository.create(sample_library)
        assert repository.exists(sample_library.id)
        
        repository.delete(sample_library.id)
        assert not repository.exists(sample_library.id)
    
    def test_clear(self, repository: InMemoryLibraryRepository):
        """Test clearing all libraries."""
        # Create some libraries
        for i in range(3):
            library = Library.create(name=f"Library {i}")
            repository.create(library)
        
        assert repository.count_all() == 3
        
        repository.clear()
        
        assert repository.count_all() == 0
        assert repository.list_all() == []
    
    def test_count_all(self, repository: InMemoryLibraryRepository):
        """Test counting libraries."""
        assert repository.count_all() == 0
        
        library1 = Library.create(name="Library 1")
        repository.create(library1)
        assert repository.count_all() == 1
        
        library2 = Library.create(name="Library 2")
        repository.create(library2)
        assert repository.count_all() == 2
        
        repository.delete(library1.id)
        assert repository.count_all() == 1
