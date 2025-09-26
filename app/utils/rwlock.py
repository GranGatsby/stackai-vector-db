"""Reader-Writer lock implementation for concurrent access control.

Provides thread-safe reader-writer lock allowing multiple concurrent readers
or single exclusive writer, preventing data races in shared data structures.
"""

import threading
from collections.abc import Generator
from contextlib import contextmanager


class RWLock:
    """Reader-writer lock with writer priority to prevent starvation.
    
    Key properties:
    - Multiple readers can acquire the lock simultaneously
    - Only one writer can hold the lock at a time
    - Writers block all readers and other writers
    - Writers have priority over new readers to prevent starvation
    """

    def __init__(self) -> None:
        """Initialize reader-writer lock."""
        self._read_ready = threading.Condition(threading.RLock())
        self._readers = 0
        self._writers_waiting = 0
        self._writer_active = False

    @contextmanager
    def read_lock(self) -> Generator[None, None, None]:
        """Context manager for acquiring a read lock.

        Usage:
            with rwlock.read_lock():
                # perform read operations
                pass
        """
        self._acquire_read()
        try:
            yield
        finally:
            self._release_read()

    @contextmanager
    def write_lock(self) -> Generator[None, None, None]:
        """Context manager for acquiring a write lock.

        Usage:
            with rwlock.write_lock():
                # perform write operations
                pass
        """
        self._acquire_write()
        try:
            yield
        finally:
            self._release_write()

    def acquire_read(self) -> None:
        """Acquire read lock (manual - prefer read_lock() context manager)."""
        self._acquire_read()

    def release_read(self) -> None:
        """Release read lock (manual - prefer read_lock() context manager)."""
        self._release_read()

    def acquire_write(self) -> None:
        """Acquire write lock (manual - prefer write_lock() context manager)."""
        self._acquire_write()

    def release_write(self) -> None:
        """Release write lock (manual - prefer write_lock() context manager)."""
        self._release_write()

    def _acquire_read(self) -> None:
        """Internal method to acquire read lock."""
        with self._read_ready:
            # Wait while there's an active writer or writers waiting
            while self._writer_active or self._writers_waiting > 0:
                self._read_ready.wait()
            self._readers += 1

    def _release_read(self) -> None:
        """Internal method to release read lock."""
        with self._read_ready:
            self._readers -= 1
            if self._readers == 0:
                # Last reader leaving, notify waiting writers
                self._read_ready.notify_all()

    def _acquire_write(self) -> None:
        """Internal method to acquire write lock."""
        with self._read_ready:
            self._writers_waiting += 1
            try:
                # Wait while there are active readers or another writer
                while self._readers > 0 or self._writer_active:
                    self._read_ready.wait()
                self._writer_active = True
            finally:
                self._writers_waiting -= 1

    def _release_write(self) -> None:
        """Internal method to release write lock."""
        with self._read_ready:
            self._writer_active = False
            # Notify all waiting readers and writers
            self._read_ready.notify_all()

    @property
    def reader_count(self) -> int:
        """Get current number of active readers."""
        with self._read_ready:
            return self._readers

    @property
    def writer_waiting_count(self) -> int:
        """Get current number of waiting writers."""
        with self._read_ready:
            return self._writers_waiting

    @property
    def writer_active(self) -> bool:
        """Check if a writer is currently active."""
        with self._read_ready:
            return self._writer_active
