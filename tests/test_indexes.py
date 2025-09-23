"""Essential tests for vector indexing algorithms.

This module contains critical tests to verify the correctness and basic
functionality of the vector indexing implementations.
"""

import numpy as np
import pytest

from app.indexes import (
    DimensionMismatchError,
    IVFIndex,
    IndexNotBuiltError,
    KDTreeIndex,
    LinearScanIndex,
    create_index,
)


class TestIndexCreation:
    """Test index creation and factory functions."""

    def test_create_linear_index(self):
        """Test creating linear scan index."""
        index = create_index("linear", dimension=3)
        assert isinstance(index, LinearScanIndex)
        
    def test_create_kdtree_index(self):
        """Test creating KD-tree index."""
        index = create_index("kdtree", dimension=3)
        assert isinstance(index, KDTreeIndex)
        
    def test_create_ivf_index(self):
        """Test creating IVF index."""
        index = create_index("ivf", dimension=3)
        assert isinstance(index, IVFIndex)
        
    def test_create_invalid_index_type(self):
        """Test error on invalid index type."""
        with pytest.raises(ValueError, match="Unsupported index type"):
            create_index("invalid_type")


class TestIndexBasicFunctionality:
    """Test basic functionality common to all index types."""

    @pytest.fixture(params=["linear", "kdtree", "ivf"])
    def index(self, request):
        """Create index instance for each type."""
        return create_index(request.param, dimension=3)

    @pytest.fixture
    def sample_vectors(self):
        """Create sample vectors for testing."""
        return [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], 
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
        ]

    def test_index_properties_before_build(self, index):
        """Test index properties before building."""
        assert index.size == 0
        assert not index.is_built

    def test_build_index_success(self, index, sample_vectors):
        """Test successful index building."""
        index.build(sample_vectors)
        
        assert index.is_built
        assert index.size == len(sample_vectors)
        assert index.dim == 3

    def test_build_empty_vectors_fails(self, index):
        """Test that building with empty vectors fails."""
        with pytest.raises(ValueError, match="empty vector collection"):
            index.build([])

    def test_query_before_build_fails(self, index):
        """Test that querying before building fails."""
        with pytest.raises(IndexNotBuiltError):
            index.query([1.0, 0.0, 0.0])

    def test_dimension_mismatch_fails(self, index, sample_vectors):
        """Test that dimension mismatch fails."""
        index.build(sample_vectors)
        
        # Query with wrong dimension
        with pytest.raises(DimensionMismatchError):
            index.query([1.0, 0.0])  # 2D instead of 3D

    def test_basic_query_functionality(self, index, sample_vectors):
        """Test basic query functionality."""
        index.build(sample_vectors)
        
        # Query with exact match
        results = index.query([1.0, 0.0, 0.0], k=1)
        
        assert len(results) == 1
        assert results[0][0] == 0  # Should find first vector
        assert results[0][1] == 0.0  # Distance should be 0


class TestLinearScanSpecific:
    """Test LinearScan-specific functionality."""

    def test_exact_results(self):
        """Test that LinearScan returns exact results."""
        vectors = [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
            [0.8, 0.2],
        ]
        
        index = LinearScanIndex()
        index.build(vectors)
        
        # Query should return results in exact distance order
        results = index.query([0.6, 0.4], k=4)
        
        # Manually verify distances are in ascending order
        distances = [result[1] for result in results]
        assert distances == sorted(distances)

    def test_incremental_updates_no_rebuild(self):
        """Test that LinearScan handles incremental updates without rebuild."""
        vectors = [[1.0, 0.0], [0.0, 1.0]]
        
        index = LinearScanIndex()
        index.build(vectors)
        
        # Add vector - should not require rebuild
        new_idx = index.add_vector([0.5, 0.5])
        assert index.is_built  # Should still be built
        assert new_idx == 2
        
        # Should be able to query immediately
        results = index.query([0.5, 0.5], k=1)
        assert results[0][0] == 2  # Should find the new vector


class TestKDTreeSpecific:
    """Test KDTree-specific functionality."""

    def test_tree_building_with_leaf_size(self):
        """Test KDTree building with different leaf sizes."""
        vectors = [[i, 0] for i in range(10)]  # 10 2D vectors
        
        index = KDTreeIndex(leaf_size=3)
        index.build(vectors)
        
        assert index.is_built
        assert index.get_tree_depth() > 1  # Should create a tree structure

    def test_query_correctness_small_dataset(self):
        """Test KDTree query correctness on small dataset."""
        vectors = [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
        
        index = KDTreeIndex()
        index.build(vectors)
        
        # Query for point closest to [0.1, 0.1]
        results = index.query([0.1, 0.1], k=1)
        
        # Should find [0.0, 0.0] as nearest
        assert results[0][0] == 0


class TestIVFSpecific:
    """Test IVF-specific functionality."""

    def test_clustering_with_sufficient_data(self):
        """Test IVF clustering with sufficient data."""
        # Create clustered data
        vectors = []
        # Cluster 1: around [0, 0]
        for i in range(10):
            vectors.append([np.random.normal(0, 0.1), np.random.normal(0, 0.1)])
        # Cluster 2: around [5, 5]  
        for i in range(10):
            vectors.append([np.random.normal(5, 0.1), np.random.normal(5, 0.1)])
        
        index = IVFIndex(n_clusters=2)
        index.build(vectors)
        
        assert index.is_built
        stats = index.get_stats()
        assert stats["n_clusters"] == 2

    def test_nprobe_parameter(self):
        """Test IVF nprobe parameter."""
        vectors = [[i, 0] for i in range(20)]
        
        index = IVFIndex(n_clusters=4, nprobe=2)
        index.build(vectors)
        
        # Change nprobe
        index.set_nprobe(3)
        
        # Should still be able to query
        results = index.query([10.0, 0.0], k=3)
        assert len(results) <= 3

    def test_incremental_updates_efficient(self):
        """Test that IVF handles incremental updates efficiently."""
        vectors = [[i, 0] for i in range(10)]
        
        index = IVFIndex(n_clusters=2)
        index.build(vectors)
        
        # Add vector - should not require full rebuild
        new_idx = index.add_vector([10.0, 0.0])
        assert index.is_built  # Should still be built
        assert new_idx == 10


class TestIndexStats:
    """Test index statistics and memory usage."""

    def test_get_stats_all_types(self):
        """Test getting statistics from all index types."""
        vectors = [[i, 0, 0] for i in range(5)]
        
        for index_type in ["linear", "kdtree", "ivf"]:
            index = create_index(index_type, dimension=3)
            index.build(vectors)
            
            stats = index.get_stats()
            
            # All indexes should report basic stats
            assert "algorithm" in stats
            assert "dimension" in stats
            assert "total_vectors" in stats
            assert "is_built" in stats
            assert "complexity" in stats

    def test_memory_usage_reporting(self):
        """Test memory usage reporting."""
        vectors = [[i, 0] for i in range(10)]
        
        index = LinearScanIndex()
        index.build(vectors)
        
        if hasattr(index, 'get_memory_usage'):
            memory_stats = index.get_memory_usage()
            assert "total" in memory_stats
            assert memory_stats["total"] > 0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_vector_index(self):
        """Test indexing a single vector."""
        vectors = [[1.0, 2.0, 3.0]]
        
        index = LinearScanIndex()
        index.build(vectors)
        
        results = index.query([1.0, 2.0, 3.0], k=1)
        assert len(results) == 1
        assert results[0][0] == 0
        assert results[0][1] == 0.0

    def test_query_k_larger_than_dataset(self):
        """Test querying for more neighbors than available."""
        vectors = [[1.0, 0.0], [0.0, 1.0]]
        
        index = LinearScanIndex()
        index.build(vectors)
        
        # Ask for more neighbors than available
        results = index.query([0.5, 0.5], k=10)
        
        # Should return only available vectors
        assert len(results) == 2

    def test_inconsistent_vector_dimensions(self):
        """Test error on inconsistent vector dimensions."""
        vectors = [
            [1.0, 0.0],      # 2D
            [0.0, 1.0, 0.0], # 3D - inconsistent!
        ]
        
        index = LinearScanIndex()
        
        with pytest.raises(DimensionMismatchError):
            index.build(vectors)
