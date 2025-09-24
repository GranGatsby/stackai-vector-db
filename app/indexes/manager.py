"""Index management with thread-safety and factory functions.

This module provides thread-safe index management and factory functions
for creating appropriate index instances based on configuration and data
characteristics.
"""

import logging
from collections.abc import Sequence

from app.core.config import settings
from app.utils import RWLock

from .base import VectorIndex
from .ivf import IVFIndex
from .kdtree import KDTreeIndex
from .linear import LinearScanIndex

logger = logging.getLogger(__name__)


def create_index(
    index_type: str = None, dimension: int = None, **kwargs
) -> VectorIndex:
    """Factory function to create vector index instances.

    Args:
        index_type: Type of index to create ("linear", "kdtree", "ivf")
        dimension: Expected vector dimension
        **kwargs: Additional arguments passed to index constructor

    Returns:
        Vector index instance

    Raises:
        ValueError: If index_type is not supported
    """
    if index_type is None:
        index_type = settings.default_index_type

    index_type = index_type.lower().strip()

    if index_type == "linear":
        return LinearScanIndex(dimension=dimension, **kwargs)
    elif index_type == "kdtree":
        return KDTreeIndex(dimension=dimension, **kwargs)
    elif index_type == "ivf":
        return IVFIndex(dimension=dimension, **kwargs)
    else:
        raise ValueError(
            f"Unsupported index type: {index_type}. "
            f"Supported types: linear, kdtree, ivf"
        )


def recommend_index_type(
    n_vectors: int, dimension: int, accuracy_priority: bool = True
) -> str:
    """Recommend the best index type based on data characteristics.

    Args:
        n_vectors: Number of vectors to index
        dimension: Vector dimension
        accuracy_priority: Whether to prioritize accuracy over speed

    Returns:
        Recommended index type ("linear", "kdtree", or "ivf")
    """
    # Small datasets: linear scan is fine and gives exact results
    if n_vectors < 1000:
        return "linear"

    # Low dimensions: KD-Tree works well
    if dimension <= 20 and n_vectors < 50000:
        return "kdtree"

    # Large datasets or high dimensions: IVF scales better
    if n_vectors >= 10000 or dimension > 50:
        return "ivf"

    # Medium-sized datasets: choose based on accuracy priority
    if accuracy_priority:
        return "kdtree" if dimension <= 20 else "linear"
    else:
        return "ivf"


class ThreadSafeIndex:
    """Thread-safe wrapper for vector indexes.

    This wrapper provides thread-safe access to vector indexes using
    read-write locks. Multiple readers can access the index concurrently,
    but writes are exclusive.
    """

    def __init__(self, index: VectorIndex) -> None:
        """Initialize the thread-safe index wrapper.

        Args:
            index: The underlying vector index to wrap
        """
        self._index = index
        self._lock = RWLock()

        logger.debug(f"Created ThreadSafeIndex wrapper for {type(index).__name__}")

    def _acquire_read(self):
        """Acquire read lock."""
        self._lock.acquire_read()

    def _release_read(self):
        """Release read lock."""
        self._lock.release_read()

    def _acquire_write(self):
        """Acquire write lock."""
        self._lock.acquire_write()

    def _release_write(self):
        """Release write lock."""
        self._lock.release_write()

    @property
    def dim(self) -> int:
        """Get vector dimension (thread-safe read)."""
        self._acquire_read()
        try:
            return self._index.dim
        finally:
            self._release_read()

    @property
    def size(self) -> int:
        """Get number of vectors (thread-safe read)."""
        self._acquire_read()
        try:
            return self._index.size
        finally:
            self._release_read()

    @property
    def is_built(self) -> bool:
        """Check if index is built (thread-safe read)."""
        self._acquire_read()
        try:
            return self._index.is_built
        finally:
            self._release_read()

    def build(self, vectors: Sequence[Sequence[float]]) -> None:
        """Build index (thread-safe write)."""
        self._acquire_write()
        try:
            self._index.build(vectors)
        finally:
            self._release_write()

    def query(
        self, query_vector: Sequence[float], k: int = 10
    ) -> list[tuple[int, float]]:
        """Query index (thread-safe read)."""
        self._acquire_read()
        try:
            return self._index.query(query_vector, k)
        finally:
            self._release_read()

    def add_vector(self, vector: Sequence[float]) -> int:
        """Add vector (thread-safe write)."""
        self._acquire_write()
        try:
            return self._index.add_vector(vector)
        finally:
            self._release_write()

    def remove_vector(self, index: int) -> bool:
        """Remove vector (thread-safe write)."""
        self._acquire_write()
        try:
            return self._index.remove_vector(index)
        finally:
            self._release_write()

    def get_stats(self) -> dict:
        """Get index statistics (thread-safe read)."""
        self._acquire_read()
        try:
            if hasattr(self._index, "get_stats"):
                return self._index.get_stats()
            return {"algorithm": type(self._index).__name__}
        finally:
            self._release_read()


class IndexManager:
    """Manager for vector indexes with automatic type selection and thread-safety.

    This class provides high-level index management including automatic
    index type selection, thread-safe operations, and index lifecycle management.
    """

    def __init__(
        self,
        index_type: str = None,
        dimension: int = None,
        thread_safe: bool = True,
        **index_kwargs,
    ) -> None:
        """Initialize the index manager.

        Args:
            index_type: Type of index to use (auto-selected if None)
            dimension: Expected vector dimension
            thread_safe: Whether to use thread-safe wrapper
            **index_kwargs: Additional arguments for index creation
        """
        self._index_type = index_type
        self._dimension = dimension
        self._thread_safe = thread_safe
        self._index_kwargs = index_kwargs
        self._index: VectorIndex | None = None

        logger.debug(
            f"Initialized IndexManager with type={index_type}, thread_safe={thread_safe}"
        )

    def _create_index(self, n_vectors: int = None) -> VectorIndex:
        """Create the appropriate index instance.

        Args:
            n_vectors: Number of vectors (for auto-selection)

        Returns:
            Vector index instance
        """
        index_type = self._index_type

        # Auto-select index type if not specified
        if index_type is None and n_vectors is not None and self._dimension is not None:
            index_type = recommend_index_type(n_vectors, self._dimension)
            logger.info(f"Auto-selected index type: {index_type}")

        # Create the index
        index = create_index(
            index_type=index_type, dimension=self._dimension, **self._index_kwargs
        )

        # Wrap with thread-safety if requested
        if self._thread_safe:
            index = ThreadSafeIndex(index)

        return index

    def build_index(self, vectors: Sequence[Sequence[float]]) -> None:
        """Build the index from vectors.

        Args:
            vectors: Vectors to index
        """
        if not vectors:
            raise ValueError("Cannot build index from empty vector collection")

        # Create index if not exists
        if self._index is None:
            self._index = self._create_index(len(vectors))

        # Build the index
        self._index.build(vectors)

        logger.info(f"Built index with {len(vectors)} vectors")

    def query(
        self, query_vector: Sequence[float], k: int = 10
    ) -> list[tuple[int, float]]:
        """Query the index for nearest neighbors.

        Args:
            query_vector: Query vector
            k: Number of neighbors to return

        Returns:
            List of (index, distance) tuples

        Raises:
            RuntimeError: If index hasn't been built
        """
        if self._index is None:
            raise RuntimeError("Index hasn't been created yet")

        return self._index.query(query_vector, k)

    def add_vector(self, vector: Sequence[float]) -> int:
        """Add a vector to the index.

        Args:
            vector: Vector to add

        Returns:
            Index of the added vector

        Raises:
            RuntimeError: If index hasn't been built
        """
        if self._index is None:
            # Create index for single vector
            self._index = self._create_index()

        return self._index.add_vector(vector)

    def remove_vector(self, index: int) -> bool:
        """Remove a vector from the index.

        Args:
            index: Index of vector to remove

        Returns:
            True if removed, False if index was invalid
        """
        if self._index is None:
            return False

        return self._index.remove_vector(index)

    def get_stats(self) -> dict:
        """Get index statistics.

        Returns:
            Dictionary with index statistics
        """
        if self._index is None:
            return {"status": "not_created"}

        stats = self._index.get_stats() if hasattr(self._index, "get_stats") else {}
        stats["manager_config"] = {
            "index_type": self._index_type,
            "dimension": self._dimension,
            "thread_safe": self._thread_safe,
        }

        return stats

    @property
    def is_built(self) -> bool:
        """Check if index is built."""
        return self._index is not None and self._index.is_built

    @property
    def size(self) -> int:
        """Get number of vectors in index."""
        return self._index.size if self._index is not None else 0
