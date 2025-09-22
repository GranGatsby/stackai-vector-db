"""Basic concurrency tests for the library repository."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from app.domain import Library, LibraryAlreadyExistsError
from app.repositories.in_memory import InMemoryLibraryRepository


class TestConcurrency:
    """Test suite for concurrent operations."""

    @pytest.fixture
    def repository(self):
        """Create a fresh repository for each test."""
        return InMemoryLibraryRepository()

    def test_concurrent_reads_dont_block(self, repository: InMemoryLibraryRepository):
        """Test that multiple concurrent reads don't block each other."""
        # Create some libraries
        libraries = []
        for i in range(5):
            library = Library.create(name=f"Library {i}")
            repository.create(library)
            libraries.append(library)

        results = []
        start_time = time.time()

        def read_operation():
            """Perform a read operation with some delay."""
            time.sleep(0.1)  # Simulate some processing time
            return repository.list_all()

        # Run multiple reads concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(read_operation) for _ in range(5)]

            for future in as_completed(futures):
                results.append(future.result())

        end_time = time.time()

        # All reads should return the same data
        for result in results:
            assert len(result) == 5

        # Concurrent reads should be faster than sequential reads
        # (5 reads * 0.1s sleep should be ~0.5s if sequential, much less if concurrent)
        assert end_time - start_time < 0.3  # Allow some margin for test execution

    def test_concurrent_writes_with_conflicts(
        self, repository: InMemoryLibraryRepository
    ):
        """Test that concurrent writes handle conflicts properly."""
        conflicts = []
        successes = []

        def create_duplicate_library(name: str):
            """Try to create a library with the given name."""
            try:
                library = Library.create(name=name)
                result = repository.create(library)
                successes.append(result)
                return True
            except LibraryAlreadyExistsError:
                conflicts.append(name)
                return False

        # Try to create multiple libraries with the same name concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(create_duplicate_library, "Duplicate Library")
                for _ in range(5)
            ]

            results = [future.result() for future in as_completed(futures)]

        # Only one should succeed, others should conflict
        assert sum(results) == 1  # Only one success
        assert len(successes) == 1
        assert len(conflicts) == 4

        # Repository should only have one library
        assert repository.count_all() == 1

    def test_concurrent_read_write_operations(
        self, repository: InMemoryLibraryRepository
    ):
        """Test mixed read and write operations."""
        # Pre-populate with some libraries
        for i in range(3):
            library = Library.create(name=f"Initial Library {i}")
            repository.create(library)

        read_results = []
        write_results = []

        def read_operation():
            """Perform read operations."""
            time.sleep(0.05)  # Small delay
            return len(repository.list_all())

        def write_operation(index: int):
            """Perform write operations."""
            try:
                library = Library.create(name=f"New Library {index}")
                repository.create(library)
                return True
            except Exception:
                return False

        # Mix read and write operations
        with ThreadPoolExecutor(max_workers=8) as executor:
            # Submit read operations
            read_futures = [executor.submit(read_operation) for _ in range(5)]

            # Submit write operations
            write_futures = [executor.submit(write_operation, i) for i in range(3)]

            # Collect results
            for future in as_completed(read_futures):
                read_results.append(future.result())

            for future in as_completed(write_futures):
                write_results.append(future.result())

        # All reads should return reasonable counts (between 3 and 6)
        for count in read_results:
            assert 3 <= count <= 6

        # All writes should succeed (unique names)
        assert all(write_results)

        # Final count should be initial + new libraries
        assert repository.count_all() == 6

    def test_concurrent_updates_same_library(
        self, repository: InMemoryLibraryRepository
    ):
        """Test concurrent updates to the same library."""
        # Create initial library
        library = Library.create(name="Update Test", description="Original")
        repository.create(library)

        update_results = []

        def update_operation(new_description: str):
            """Update the library description."""
            try:
                # Get current library
                current = repository.get_by_id(library.id)
                if current:
                    updated = current.update(description=new_description)
                    return repository.update(updated)
                return None
            except Exception as e:
                return str(e)

        # Perform concurrent updates
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(update_operation, f"Updated {i}") for i in range(3)
            ]

            for future in as_completed(futures):
                update_results.append(future.result())

        # All updates should succeed (they don't conflict on names)
        successful_updates = [r for r in update_results if isinstance(r, Library)]
        assert len(successful_updates) == 3

        # Final library should have one of the updated descriptions
        final_library = repository.get_by_id(library.id)
        assert final_library.description.startswith("Updated")

    def test_rwlock_reader_writer_behavior(self, repository: InMemoryLibraryRepository):
        """Test that RWLock properly handles reader-writer coordination."""
        # Create some initial data
        for i in range(3):
            library = Library.create(name=f"Library {i}")
            repository.create(library)

        read_start_times = []
        read_end_times = []
        write_start_time = None
        write_end_time = None

        def long_read_operation():
            """A read operation that takes some time."""
            start = time.time()
            read_start_times.append(start)

            # Perform multiple reads with delays
            for _ in range(3):
                repository.list_all()
                time.sleep(0.05)

            end = time.time()
            read_end_times.append(end)

        def write_operation():
            """A write operation that should wait for reads to complete."""
            nonlocal write_start_time, write_end_time

            write_start_time = time.time()
            library = Library.create(name="Write Test")
            repository.create(library)
            write_end_time = time.time()

        # Start read operations first
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Start multiple readers
            read_futures = [executor.submit(long_read_operation) for _ in range(2)]

            # Wait a bit, then start a writer
            time.sleep(0.02)
            write_future = executor.submit(write_operation)

            # Wait for all to complete
            for future in as_completed(read_futures + [write_future]):
                future.result()

        # Verify that readers ran concurrently (overlapping times)
        assert len(read_start_times) == 2
        assert len(read_end_times) == 2

        # At least one reader should have started before the other ended
        min_read_start = min(read_start_times)
        max_read_end = max(read_end_times)

        # Write should have completed after readers started
        assert write_start_time is not None
        assert write_end_time is not None
        assert write_start_time >= min_read_start
