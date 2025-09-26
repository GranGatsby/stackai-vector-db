"""IVF (Inverted File) vector index implementation.

This module implements an IVF (Inverted File) index, which uses coarse quantization
to partition vectors into clusters and maintains inverted lists for efficient
approximate nearest neighbor search. This approach scales well to large datasets.
"""

import logging
from collections.abc import Sequence

import numpy as np

from .base import BaseVectorIndex, DimensionMismatchError

logger = logging.getLogger(__name__)


class IVFIndex(BaseVectorIndex):
    """IVF (Inverted File) vector index implementation.

    IVF uses coarse quantization to partition the vector space into clusters.
    Each cluster has an associated inverted list containing the vectors assigned
    to that cluster. During search, only a subset of clusters is examined,
    providing significant speedup for large datasets.

    Time Complexity:
    - Build: O(N * C * I) - N vectors, C clusters, I k-means iterations
    - Query: O(P * M + k) - P probes, M avg vectors per cluster
    - Add: O(C) - Find nearest cluster and add to inverted list
    - Remove: O(M) - Find and remove from inverted list

    Space Complexity:
    - O(N + C * D) - N vectors plus C cluster centroids

    Characteristics:
    - Excellent scalability for large datasets
    - Approximate results (tunable via nprobe parameter)
    - Good performance in high dimensions
    - Memory efficient for large datasets
    - Fast incremental updates

    Best Use Cases:
    - Large datasets (>10K vectors)
    - High-dimensional embeddings
    - When approximate results are acceptable
    - Applications requiring fast incremental updates
    """

    def __init__(
        self,
        dimension: int | None = None,
        n_clusters: int | None = None,
        nprobe: int = 1,
        max_kmeans_iter: int = 50,
    ) -> None:
        """Initialize the IVF index.

        Args:
            dimension: Expected vector dimension (auto-detected if None)
            n_clusters: Number of clusters for quantization (auto-determined if None)
            nprobe: Number of clusters to probe during search
            max_kmeans_iter: Maximum iterations for k-means clustering
        """
        super().__init__(dimension)

        self._n_clusters = n_clusters
        self._nprobe = max(1, nprobe)
        self._max_kmeans_iter = max_kmeans_iter

        # Will be initialized during build
        self._centroids: np.ndarray | None = None
        self._inverted_lists: list[list[tuple[int, np.ndarray]]] = []

        logger.debug(
            f"Initialized IVFIndex with n_clusters={n_clusters}, nprobe={nprobe}"
        )

    def _determine_n_clusters(self, n_vectors: int) -> int:
        """Determine the optimal number of clusters based on dataset size.

        Args:
            n_vectors: Number of vectors in the dataset

        Returns:
            Optimal number of clusters
        """
        if self._n_clusters is not None:
            return self._n_clusters

        # Heuristic: roughly sqrt(N/2) clusters, with reasonable bounds
        n_clusters = max(1, min(1000, int(np.sqrt(n_vectors / 2))))

        logger.debug(f"Auto-determined n_clusters={n_clusters} for {n_vectors} vectors")
        return n_clusters

    def _build_index(self) -> None:
        """Build the IVF index using k-means clustering."""
        # Filter out removed vectors
        active_vectors = [
            (i, vector) for i, vector in enumerate(self._vectors) if vector is not None
        ]

        if not active_vectors:
            self._centroids = None
            self._inverted_lists = []
            return

        vectors = np.array([vector for _, vector in active_vectors])
        indices = [i for i, _ in active_vectors]

        # Determine number of clusters
        n_clusters = self._determine_n_clusters(len(vectors))
        n_clusters = min(
            n_clusters, len(vectors)
        )  # Can't have more clusters than vectors

        # Perform k-means clustering
        self._centroids = self._kmeans_clustering(vectors, n_clusters)

        # Initialize inverted lists
        self._inverted_lists = [[] for _ in range(n_clusters)]

        # Assign vectors to clusters
        for original_idx, vector in zip(indices, vectors, strict=False):
            cluster_id = self._find_nearest_cluster(vector)
            self._inverted_lists[cluster_id].append((original_idx, vector))

        # Log cluster statistics
        cluster_sizes = [len(inv_list) for inv_list in self._inverted_lists]
        logger.info(
            f"Built IVF index: {n_clusters} clusters, "
            f"avg size: {np.mean(cluster_sizes):.1f}, "
            f"min/max: {min(cluster_sizes)}/{max(cluster_sizes)}"
        )

    def _kmeans_clustering(self, vectors: np.ndarray, n_clusters: int) -> np.ndarray:
        """Perform k-means clustering to find cluster centroids.

        Args:
            vectors: Array of vectors to cluster
            n_clusters: Number of clusters

        Returns:
            Array of cluster centroids
        """
        n_vectors, _dim = vectors.shape

        # Initialize centroids randomly
        rng = np.random.RandomState(42)  # Fixed seed for reproducibility
        centroid_indices = rng.choice(n_vectors, n_clusters, replace=False)
        centroids = vectors[centroid_indices].copy()

        for iteration in range(self._max_kmeans_iter):
            # Assign vectors to nearest centroids
            assignments = np.zeros(n_vectors, dtype=int)
            for i, vector in enumerate(vectors):
                distances = [
                    self.euclidean_distance(vector, centroid) for centroid in centroids
                ]
                assignments[i] = np.argmin(distances)

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for cluster_id in range(n_clusters):
                cluster_vectors = vectors[assignments == cluster_id]
                if len(cluster_vectors) > 0:
                    new_centroids[cluster_id] = np.mean(cluster_vectors, axis=0)
                else:
                    # Keep old centroid if no vectors assigned
                    new_centroids[cluster_id] = centroids[cluster_id]

            # Check for convergence
            centroid_shift = np.mean(
                [
                    self.euclidean_distance(old, new)
                    for old, new in zip(centroids, new_centroids, strict=False)
                ]
            )

            centroids = new_centroids

            if centroid_shift < 1e-6:
                logger.debug(f"K-means converged after {iteration + 1} iterations")
                break

        return centroids

    def _find_nearest_cluster(self, vector: np.ndarray) -> int:
        """Find the nearest cluster centroid for a vector.

        Args:
            vector: Vector to find nearest cluster for

        Returns:
            Index of the nearest cluster
        """
        if self._centroids is None:
            return 0

        distances = [
            self.euclidean_distance(vector, centroid) for centroid in self._centroids
        ]
        return int(np.argmin(distances))

    def _query_index(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Perform IVF query to find k nearest neighbors.

        Args:
            query_vector: Query vector as numpy array
            k: Number of neighbors to find

        Returns:
            List of (index, distance) tuples for k nearest neighbors
        """
        if self._centroids is None or not self._inverted_lists:
            return []

        # Find the nprobe nearest clusters to search
        cluster_distances = [
            (i, self.euclidean_distance(query_vector, centroid))
            for i, centroid in enumerate(self._centroids)
        ]
        cluster_distances.sort(key=lambda x: x[1])

        clusters_to_probe = [
            cluster_id for cluster_id, _ in cluster_distances[: self._nprobe]
        ]

        # Search the selected clusters
        candidates = []
        for cluster_id in clusters_to_probe:
            for original_idx, vector in self._inverted_lists[cluster_id]:
                distance = self.euclidean_distance(query_vector, vector)
                candidates.append((original_idx, distance))

        # Sort by distance and return top k
        candidates.sort(key=lambda x: x[1])
        return candidates[:k]

    def add_vector(self, vector: Sequence[float]) -> int:
        """Add a vector to the IVF index.

        For IVF, adding a vector is efficient as we just need to find the
        nearest cluster and add to its inverted list.

        Args:
            vector: Vector to add

        Returns:
            Index position of the added vector
        """
        np_vector = np.array(vector, dtype=np.float32)

        # Validate dimension
        if self._dimension is not None and len(np_vector) != self._dimension:
            raise DimensionMismatchError(
                f"Vector dimension {len(np_vector)} doesn't match index dimension {self._dimension}"
            )

        if self._dimension is None:
            self._dimension = len(np_vector)

        # Add to vector list
        self._vectors.append(np_vector)
        index = self._size
        self._size += 1

        # If index is built, add to appropriate inverted list
        if self._is_built and self._centroids is not None:
            cluster_id = self._find_nearest_cluster(np_vector)
            self._inverted_lists[cluster_id].append((index, np_vector))
        else:
            # Mark as needing rebuild
            self._is_built = False

        logger.debug(f"Added vector at index {index}")
        return index

    def remove_vector(self, index: int) -> bool:
        """Remove a vector from the IVF index.

        Args:
            index: Index position of vector to remove

        Returns:
            True if vector was removed, False if index was invalid
        """
        if index < 0 or index >= self._size or self._vectors[index] is None:
            return False

        # Remove from inverted lists if index is built
        if self._is_built and self._inverted_lists:
            for inv_list in self._inverted_lists:
                inv_list[:] = [(i, v) for i, v in inv_list if i != index]

        # Mark vector as removed
        self._vectors[index] = None

        logger.debug(f"Removed vector at index {index}")
        return True

    def set_nprobe(self, nprobe: int) -> None:
        """Set the number of clusters to probe during search.

        Args:
            nprobe: Number of clusters to probe (higher = more accurate but slower)
        """
        if self._centroids is not None:
            # Limit nprobe to the actual number of clusters
            max_nprobe = len(self._centroids)
            self._nprobe = max(1, min(nprobe, max_nprobe))
        else:
            # If index not built yet, just ensure it's >= 1
            self._nprobe = max(1, nprobe)

        logger.debug(f"Set nprobe to {self._nprobe}")

    def get_memory_usage(self) -> dict[str, int]:
        """Get memory usage statistics for the IVF index.

        Returns:
            Dictionary with memory usage information in bytes
        """
        vector_memory = sum(
            vector.nbytes for vector in self._vectors if vector is not None
        )

        # Centroid memory
        centroid_memory = self._centroids.nbytes if self._centroids is not None else 0

        # Inverted list overhead (rough estimate)
        inv_list_overhead = len(self._inverted_lists) * 64  # Rough estimate per list

        return {
            "vectors": vector_memory,
            "centroids": centroid_memory,
            "inverted_lists": inv_list_overhead,
            "total": vector_memory + centroid_memory + inv_list_overhead,
        }

    def get_stats(self) -> dict[str, any]:
        """Get comprehensive statistics about the IVF index.

        Returns:
            Dictionary with index statistics
        """
        active_vectors = sum(1 for vector in self._vectors if vector is not None)
        removed_vectors = self.size - active_vectors

        memory_stats = self.get_memory_usage()

        cluster_stats = {}
        if self._inverted_lists:
            cluster_sizes = [len(inv_list) for inv_list in self._inverted_lists]
            cluster_stats = {
                "n_clusters": len(self._inverted_lists),
                "avg_cluster_size": np.mean(cluster_sizes) if cluster_sizes else 0,
                "min_cluster_size": min(cluster_sizes) if cluster_sizes else 0,
                "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
            }

        return {
            "algorithm": "IVF",
            "dimension": self.dim if self._dimension else None,
            "total_vectors": self.size,
            "active_vectors": active_vectors,
            "removed_vectors": removed_vectors,
            "is_built": self.is_built,
            "nprobe": self._nprobe,
            "max_kmeans_iter": self._max_kmeans_iter,
            **cluster_stats,
            "memory_usage_bytes": memory_stats,
            "complexity": {
                "build_time": "O(N * C * I)",
                "query_time": "O(P * M + k)",
                "space": "O(N + C * D)",
            },
            "characteristics": [
                "Excellent scalability for large datasets",
                "Approximate results (tunable via nprobe)",
                "Good performance in high dimensions",
                "Fast incremental updates",
            ],
        }
