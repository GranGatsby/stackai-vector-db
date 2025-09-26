"""Index management with thread-safety and factory functions."""

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
    index_type: str | None = None, dimension: int | None = None, **kwargs
) -> VectorIndex:
    """Factory to create vector index instances."""
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


# Index selection thresholds
_SMALL_DATASET_THRESHOLD = 1000
_LARGE_DATASET_THRESHOLD = 10000
_MEDIUM_DATASET_THRESHOLD = 50000
_LOW_DIMENSION_THRESHOLD = 20
_HIGH_DIMENSION_THRESHOLD = 50


def recommend_index_type(
    n_vectors: int, dimension: int, accuracy_priority: bool = True
) -> str:
    """Recommend best index type based on data characteristics."""
    if n_vectors < _SMALL_DATASET_THRESHOLD:
        return "linear"

    if dimension <= _LOW_DIMENSION_THRESHOLD and n_vectors < _MEDIUM_DATASET_THRESHOLD:
        return "kdtree"

    if n_vectors >= _LARGE_DATASET_THRESHOLD or dimension > _HIGH_DIMENSION_THRESHOLD:
        return "ivf"

    if accuracy_priority:
        return "kdtree" if dimension <= _LOW_DIMENSION_THRESHOLD else "linear"
    else:
        return "ivf"


class ThreadSafeIndex:
    """Thread-safe wrapper using read-write locks."""

    def __init__(self, index: VectorIndex) -> None:
        """Initialize thread-safe wrapper."""
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
        """Vector dimension (thread-safe)."""
        self._acquire_read()
        try:
            return self._index.dim
        finally:
            self._release_read()

    @property
    def size(self) -> int:
        """Number of vectors (thread-safe)."""
        self._acquire_read()
        try:
            return self._index.size
        finally:
            self._release_read()

    @property
    def is_built(self) -> bool:
        """Index built status (thread-safe)."""
        self._acquire_read()
        try:
            return self._index.is_built
        finally:
            self._release_read()

    def build(self, vectors: Sequence[Sequence[float]]) -> None:
        """Build index (thread-safe)."""
        self._acquire_write()
        try:
            self._index.build(vectors)
        finally:
            self._release_write()

    def query(
        self, query_vector: Sequence[float], k: int = 10
    ) -> list[tuple[int, float]]:
        """Query index (thread-safe)."""
        self._acquire_read()
        try:
            return self._index.query(query_vector, k)
        finally:
            self._release_read()

    def add_vector(self, vector: Sequence[float]) -> int:
        """Add vector (thread-safe)."""
        self._acquire_write()
        try:
            return self._index.add_vector(vector)
        finally:
            self._release_write()

    def remove_vector(self, index: int) -> bool:
        """Remove vector (thread-safe)."""
        self._acquire_write()
        try:
            return self._index.remove_vector(index)
        finally:
            self._release_write()

    def get_stats(self) -> dict:
        """Index statistics (thread-safe)."""
        self._acquire_read()
        try:
            if hasattr(self._index, "get_stats"):
                return self._index.get_stats()
            return {"algorithm": type(self._index).__name__}
        finally:
            self._release_read()


class IndexManager:
    """High-level index manager with auto-selection and thread-safety."""

    def __init__(
        self,
        index_type: str | None = None,
        dimension: int | None = None,
        thread_safe: bool = True,
        **index_kwargs,
    ) -> None:
        """Initialize index manager."""
        self._index_type = index_type
        self._dimension = dimension
        self._thread_safe = thread_safe
        self._index_kwargs = index_kwargs
        self._index: VectorIndex | None = None

        logger.debug(
            f"Initialized IndexManager with type={index_type}, thread_safe={thread_safe}"
        )

    def _create_index(self, n_vectors: int | None = None) -> VectorIndex:
        """Create appropriate index instance."""
        index_type = self._index_type

        if index_type is None and n_vectors is not None and self._dimension is not None:
            index_type = recommend_index_type(n_vectors, self._dimension)
            logger.info(f"Auto-selected index type: {index_type}")

        index = create_index(
            index_type=index_type, dimension=self._dimension, **self._index_kwargs
        )

        if self._thread_safe:
            index = ThreadSafeIndex(index)

        return index

    def build_index(self, vectors: Sequence[Sequence[float]]) -> None:
        """Build index from vectors."""
        if not vectors:
            raise ValueError("Cannot build index from empty vector collection")

        if self._index is None:
            self._index = self._create_index(len(vectors))

        self._index.build(vectors)

        logger.info(f"Built index with {len(vectors)} vectors")

    def query(
        self, query_vector: Sequence[float], k: int = 10
    ) -> list[tuple[int, float]]:
        """Query index for nearest neighbors."""
        if self._index is None:
            raise RuntimeError("Index hasn't been created yet")

        return self._index.query(query_vector, k)

    def add_vector(self, vector: Sequence[float]) -> int:
        """Add vector to index."""
        if self._index is None:
            self._index = self._create_index()

        return self._index.add_vector(vector)

    def remove_vector(self, index: int) -> bool:
        """Remove vector from index."""
        if self._index is None:
            return False

        return self._index.remove_vector(index)

    def get_stats(self) -> dict:
        """Index statistics."""
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
        """True if index is built."""
        return self._index is not None and self._index.is_built

    @property
    def size(self) -> int:
        """Number of vectors in index."""
        return self._index.size if self._index is not None else 0
