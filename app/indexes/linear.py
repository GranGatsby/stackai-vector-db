"""Linear scan vector index implementation."""

import logging
from collections.abc import Sequence

import numpy as np

from .base import BaseVectorIndex

logger = logging.getLogger(__name__)


class LinearScanIndex(BaseVectorIndex):
    """Linear scan index - exhaustive search baseline.
    
    Time: Build O(N), Query O(N*D), Space O(N*D)
    Best for: Small datasets, exact results, reference implementation
    """

    def __init__(
        self, dimension: int | None = None, distance_metric: str = "euclidean"
    ) -> None:
        """Initialize linear scan index."""
        super().__init__(dimension)

        if distance_metric not in ("euclidean", "cosine"):
            raise ValueError("distance_metric must be 'euclidean' or 'cosine'")

        self._distance_metric = distance_metric
        self._distance_func = (
            self.euclidean_distance
            if distance_metric == "euclidean"
            else self.cosine_distance
        )

        logger.debug(
            f"Initialized LinearScanIndex with distance_metric={distance_metric}"
        )

    def _build_index(self) -> None:
        """Build linear scan index (trivial - just store vectors)."""
        # Vectors already stored, nothing to build
        logger.debug(f"Built LinearScanIndex with {self.size} vectors")

    def _query_index(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Linear scan query for k nearest neighbors."""
        if self.size == 0:
            return []

        distances = []

        for i, vector in enumerate(self._vectors):
            if vector is None:
                continue

            distance = self._distance_func(query_vector, vector)
            distances.append((i, distance))

        distances.sort(key=lambda x: x[1])

        return distances[:k]

    def add_vector(self, vector: Sequence[float]) -> int:
        """Add vector to linear scan index (efficient, no rebuild needed)."""
        index = super().add_vector(vector)
        self._is_built = True
        return index

    def remove_vector(self, index: int) -> bool:
        """Remove vector from linear scan index (efficient, no rebuild needed)."""
        result = super().remove_vector(index)
        if result:
            self._is_built = True
        return result

    def get_memory_usage(self) -> dict[str, int]:
        """Memory usage statistics in bytes."""
        vector_memory = sum(
            vector.nbytes for vector in self._vectors if vector is not None
        )

        return {
            "vectors": vector_memory,
            "total": vector_memory,
            "overhead": 0,
        }

    def get_stats(self) -> dict[str, any]:
        """Comprehensive linear scan statistics."""
        active_vectors = sum(1 for vector in self._vectors if vector is not None)
        removed_vectors = self.size - active_vectors

        memory_stats = self.get_memory_usage()

        return {
            "algorithm": "LinearScan",
            "dimension": self.dim if self._dimension else None,
            "total_vectors": self.size,
            "active_vectors": active_vectors,
            "removed_vectors": removed_vectors,
            "is_built": self.is_built,
            "distance_metric": self._distance_metric,
            "memory_usage_bytes": memory_stats,
            "complexity": {
                "build_time": "O(N)",
                "query_time": "O(N * D)",
                "space": "O(N * D)",
            },
        }
