"""Index service for managing vector indexes per library.

This service is responsible for maintaining the state of vector indexes for each library,
including lazy building, dirty flag management, and ensuring thread-safety through
per-library RWLocks. It handles embedding generation and dimension validation.

Key Features:
- Per-library index management with independent state
- Lazy index building with dirty flag tracking
- Thread-safe operations with per-library RWLocks
- Atomic snapshot swapping for concurrent access
- Automatic embedding generation with batch processing
- Dimension validation across all embeddings in a library

Architecture:
- Uses immutable snapshots for concurrent read access
- Implements copy-on-write semantics for index updates
- Maintains separate locks per library to avoid global contention
- Supports configurable dirty threshold for rebuild decisions
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from uuid import UUID

from app.clients import EmbeddingClient, create_embedding_client
from app.core.config import settings
from app.domain import (
    Chunk,
    EmbeddingDimensionMismatchError,
    IndexBuildError,
    IndexNotBuiltError,
    LibraryNotFoundError,
)
from app.indexes import VectorIndex, create_index
from app.repositories.ports import ChunkRepository, LibraryRepository
from app.utils import RWLock

logger = logging.getLogger(__name__)


class IndexAlgo(str, Enum):
    """Supported vector index algorithms."""
    
    LINEAR = "linear"
    KDTREE = "kdtree"
    IVF = "ivf"


@dataclass(frozen=True)
class IndexSnapshot:
    """Immutable snapshot of a vector index for thread-safe access.
    
    This class represents an immutable view of an index at a specific point in time.
    It's used to provide consistent read access while allowing atomic updates
    through snapshot swapping.
    
    Attributes:
        index: The vector index instance
        chunk_ids: List of chunk IDs corresponding to vectors in the index
        built_at: Timestamp when this snapshot was created
        version: Version number for this snapshot
        embedding_dim: Dimension of embeddings in this index
    """
    
    index: VectorIndex
    chunk_ids: list[UUID]
    built_at: float
    version: int
    embedding_dim: int

    @property
    def size(self) -> int:
        """Get the number of vectors in this snapshot."""
        return self.index.size

    @property
    def is_built(self) -> bool:
        """Check if the index in this snapshot is built."""
        return self.index.is_built


@dataclass
class IndexState:
    """Mutable state for a library's vector index.
    
    This class tracks the mutable state of an index including dirty flags,
    configuration, and the current immutable snapshot. It's protected by
    per-library RWLocks for thread-safe access.
    
    Attributes:
        library_id: UUID of the library this index belongs to
        algorithm: Index algorithm being used
        is_dirty: Whether the index needs rebuilding
        dirty_count: Number of changes since last build
        total_chunks: Total number of chunks when last built
        current_snapshot: Current immutable snapshot (None if not built)
        lock: Per-library reader-writer lock for thread safety
        embedding_dim: Expected embedding dimension (None if not determined)
    """
    
    library_id: UUID
    algorithm: IndexAlgo = IndexAlgo.LINEAR
    is_dirty: bool = True
    dirty_count: int = 0
    total_chunks: int = 0
    current_snapshot: Optional[IndexSnapshot] = None
    lock: RWLock = field(default_factory=RWLock)
    embedding_dim: Optional[int] = None

    @property
    def is_built(self) -> bool:
        """Check if this index has been built."""
        return self.current_snapshot is not None and self.current_snapshot.is_built

    @property
    def version(self) -> int:
        """Get the current version number."""
        return self.current_snapshot.version if self.current_snapshot else 0

    @property
    def built_at(self) -> Optional[float]:
        """Get the timestamp when this index was last built."""
        return self.current_snapshot.built_at if self.current_snapshot else None

    @property
    def size(self) -> int:
        """Get the number of vectors in the current index."""
        return self.current_snapshot.size if self.current_snapshot else 0

    def should_rebuild(self, rebuild_threshold: float) -> bool:
        """Determine if the index should be rebuilt based on dirty ratio.
        
        Args:
            rebuild_threshold: Fraction of changes that triggers rebuild (0.0-1.0)
            
        Returns:
            True if index should be rebuilt
        """
        if not self.is_built:
            return True
            
        if self.total_chunks == 0:
            return True
            
        dirty_ratio = self.dirty_count / max(self.total_chunks, 1)
        return dirty_ratio >= rebuild_threshold


@dataclass(frozen=True)
class IndexStatus:
    """Status information for a library's vector index.
    
    This immutable class provides a snapshot of index status information
    that can be safely returned to callers without exposing internal state.
    
    Attributes:
        library_id: UUID of the library
        algorithm: Index algorithm being used
        is_built: Whether the index has been built
        is_dirty: Whether the index needs rebuilding
        size: Number of vectors in the index
        embedding_dim: Dimension of embeddings (None if not determined)
        built_at: Timestamp when index was last built (None if not built)
        version: Version number of the current index
        dirty_count: Number of changes since last build
    """
    
    library_id: UUID
    algorithm: IndexAlgo
    is_built: bool
    is_dirty: bool
    size: int
    embedding_dim: Optional[int]
    built_at: Optional[float]
    version: int
    dirty_count: int


class IndexService:
    """Service for managing vector indexes per library.
    
    This service maintains separate vector indexes for each library, providing
    thread-safe operations through per-library RWLocks. It implements lazy
    building, dirty flag management, and atomic snapshot swapping for concurrent
    access.
    
    Key Responsibilities:
    1. Maintain index state per library with proper isolation
    2. Lazy build indexes when needed based on dirty flags
    3. Ensure all chunks have embeddings before building
    4. Validate embedding dimensions within each library
    5. Provide thread-safe access through immutable snapshots
    6. Handle atomic updates through snapshot swapping
    
    Time Complexity:
    - get_status(): O(1) - Simple state access
    - mark_dirty(): O(1) - Flag update
    - build(): O(N*D + I) where N=vectors, D=dimension, I=index-specific build cost
      - Linear: O(N*D) - Store vectors in memory
      - KDTree: O(N*D*log(N)) - Tree construction
      - IVF: O(N*D*C*K) - K-means clustering with C clusters, K iterations
    
    Thread Safety:
    - Uses per-library RWLocks to avoid global contention
    - Read operations (get_status) use read locks
    - Write operations (build, mark_dirty) use write locks
    - Serves immutable snapshots to prevent data races
    - Atomic snapshot swapping ensures consistency
    """

    def __init__(
        self,
        library_repository: LibraryRepository,
        chunk_repository: ChunkRepository,
        embedding_client: Optional[EmbeddingClient] = None,
        rebuild_threshold: Optional[float] = None,
    ) -> None:
        """Initialize the IndexService.
        
        Args:
            library_repository: Repository for library operations
            chunk_repository: Repository for chunk operations
            embedding_client: Client for generating embeddings (auto-created if None)
            rebuild_threshold: Dirty ratio threshold for rebuilds (uses config default if None)
        """
        self._library_repo = library_repository
        self._chunk_repo = chunk_repository
        self._embedding_client = embedding_client or create_embedding_client()
        self._rebuild_threshold = rebuild_threshold or settings.index_rebuild_threshold
        
        # Per-library index states - protected by individual locks
        self._index_states: dict[UUID, IndexState] = {}
        # Global lock only for managing the index_states dict itself
        self._states_lock = RWLock()
        
        logger.info(
            f"Initialized IndexService with rebuild_threshold={self._rebuild_threshold}"
        )

    def get_status(self, library_id: UUID) -> IndexStatus:
        """Get the current status of a library's index.
        
        This method provides a read-only view of the index status without
        triggering any builds or modifications. It's thread-safe and uses
        read locks for concurrent access.
        
        Args:
            library_id: UUID of the library to check
            
        Returns:
            IndexStatus with current state information
            
        Raises:
            LibraryNotFoundError: If the library doesn't exist
        """
        # Verify library exists
        if not self._library_repo.exists(library_id):
            raise LibraryNotFoundError(str(library_id))
        
        # Get or create index state
        state = self._get_or_create_state(library_id)
        
        # Use read lock for status access
        with state.lock.read_lock():
            return IndexStatus(
                library_id=library_id,
                algorithm=state.algorithm,
                is_built=state.is_built,
                is_dirty=state.is_dirty,
                size=state.size,
                embedding_dim=state.embedding_dim,
                built_at=state.built_at,
                version=state.version,
                dirty_count=state.dirty_count,
            )

    def mark_dirty(self, library_id: UUID) -> None:
        """Mark a library's index as dirty (needing rebuild).
        
        This method is called when chunks or documents are modified,
        indicating that the index needs to be rebuilt. It increments
        the dirty counter for rebuild threshold calculations.
        
        Args:
            library_id: UUID of the library to mark as dirty
            
        Raises:
            LibraryNotFoundError: If the library doesn't exist
        """
        # Verify library exists
        if not self._library_repo.exists(library_id):
            raise LibraryNotFoundError(str(library_id))
        
        # Get or create index state
        state = self._get_or_create_state(library_id)
        
        # Use write lock for state modification
        with state.lock.write_lock():
            state.is_dirty = True
            state.dirty_count += 1
            
        logger.debug(
            f"Marked index for library {library_id} as dirty "
            f"(dirty_count={state.dirty_count})"
        )

    def build(
        self, 
        library_id: UUID, 
        algorithm: Optional[IndexAlgo | str] = None
    ) -> IndexStatus:
        """Build or rebuild the vector index for a library.
        
        This method performs the complete index building process:
        1. Load all chunks for the library
        2. Ensure all chunks have embeddings (generate if missing)
        3. Validate embedding dimensions are consistent
        4. Build the vector index with the specified algorithm
        5. Create an immutable snapshot and swap it atomically
        6. Reset dirty flags and counters
        
        The build process uses write locks to ensure exclusive access
        during construction, but serves the previous snapshot to concurrent
        readers until the new one is ready.
        
        Args:
            library_id: UUID of the library to build index for
            algorithm: Index algorithm to use (str or IndexAlgo, uses current/default if None)
            
        Returns:
            IndexStatus after successful build
            
        Raises:
            LibraryNotFoundError: If the library doesn't exist
            IndexBuildError: If index building fails
            EmbeddingDimensionMismatchError: If embeddings have inconsistent dimensions
        """
        # Verify library exists
        library = self._library_repo.get_by_id(library_id)
        if library is None:
            raise LibraryNotFoundError(str(library_id))
        
        # Get or create index state
        state = self._get_or_create_state(library_id)
        
        # Use write lock for building
        with state.lock.write_lock():
            try:
                # Update algorithm if provided
                if algorithm is not None:
                    # Convert string to enum if needed
                    if isinstance(algorithm, str):
                        state.algorithm = IndexAlgo(algorithm)
                    else:
                        state.algorithm = algorithm
                
                logger.info(
                    f"Building {state.algorithm.value} index for library {library_id} "
                    f"({library.name})"
                )
                
                # Load vectors and chunk IDs
                vectors, chunk_ids = self._load_vectors_and_ids(library_id, state)
                
                if not vectors:
                    logger.info(f"No chunks found for library {library_id}, creating empty index")
                    # Set default embedding dimension for empty library
                    default_dim = settings.default_embedding_dim
                    if state.embedding_dim is None:
                        state.embedding_dim = default_dim
                    
                    # Create empty index snapshot
                    empty_index = create_index(
                        index_type=state.algorithm.value,
                        dimension=state.embedding_dim
                    )
                    
                    new_snapshot = IndexSnapshot(
                        index=empty_index,
                        chunk_ids=[],
                        built_at=time.time(),
                        version=(state.version + 1),
                        embedding_dim=state.embedding_dim,
                    )
                else:
                    # Build index with vectors
                    start_time = time.time()
                    
                    index = create_index(
                        index_type=state.algorithm.value,
                        dimension=state.embedding_dim
                    )
                    
                    # Build the index
                    index.build(vectors)
                    
                    build_time = time.time() - start_time
                    logger.info(
                        f"Built {state.algorithm.value} index for library {library_id} "
                        f"with {len(vectors)} vectors in {build_time:.2f}s"
                    )
                    
                    # Create new immutable snapshot
                    new_snapshot = IndexSnapshot(
                        index=index,
                        chunk_ids=chunk_ids,
                        built_at=time.time(),
                        version=(state.version + 1),
                        embedding_dim=state.embedding_dim,
                    )
                
                # Atomic snapshot swap
                state.current_snapshot = new_snapshot
                state.is_dirty = False
                state.dirty_count = 0
                state.total_chunks = len(chunk_ids)
                
                logger.info(
                    f"Successfully built index for library {library_id} "
                    f"(version={new_snapshot.version}, size={new_snapshot.size})"
                )
                
                # Return status
                return IndexStatus(
                    library_id=library_id,
                    algorithm=state.algorithm,
                    is_built=True,
                    is_dirty=False,
                    size=new_snapshot.size,
                    embedding_dim=state.embedding_dim,
                    built_at=new_snapshot.built_at,
                    version=new_snapshot.version,
                    dirty_count=0,
                )
                
            except Exception as e:
                error_msg = f"Failed to build index for library {library_id}: {e}"
                logger.error(error_msg)
                raise IndexBuildError(str(library_id), str(e)) from e

    def query(
        self, 
        library_id: UUID, 
        query_vector: list[float], 
        k: int = 10
    ) -> list[tuple[UUID, float]]:
        """Execute a k-NN query on a library's index.
        
        This method executes a vector similarity search on the specified library's
        index, returning the k nearest neighbors. It uses read locks to ensure
        thread-safe access to the index snapshot.
        
        Args:
            library_id: UUID of the library to query
            query_vector: Query embedding vector
            k: Number of nearest neighbors to return
            
        Returns:
            List of (chunk_id, distance) tuples ordered by distance (ascending)
            
        Raises:
            LibraryNotFoundError: If the library doesn't exist
            IndexNotBuiltError: If the index hasn't been built
            EmbeddingDimensionMismatchError: If query vector dimension doesn't match
        """
        # Verify library exists
        if not self._library_repo.exists(library_id):
            raise LibraryNotFoundError(str(library_id))
        
        # Get index state
        state = self._get_or_create_state(library_id)
        
        # Execute query with read lock
        with state.lock.read_lock():
            # Get current snapshot
            snapshot = state.current_snapshot
            if snapshot is None or not snapshot.is_built:
                raise IndexNotBuiltError(str(library_id))
            
            # Validate embedding dimension
            if len(query_vector) != snapshot.embedding_dim:
                raise EmbeddingDimensionMismatchError(
                    snapshot.embedding_dim, 
                    len(query_vector)
                )
            
            # Execute k-NN query on the index
            query_results = snapshot.index.query(query_vector, k)
            
            # Convert vector indices to chunk IDs
            chunk_results = []
            for vector_index, distance in query_results:
                if vector_index < len(snapshot.chunk_ids):
                    chunk_id = snapshot.chunk_ids[vector_index]
                    chunk_results.append((chunk_id, distance))
            
            return chunk_results

    def _get_or_create_state(self, library_id: UUID) -> IndexState:
        """Get or create index state for a library.
        
        This method ensures that each library has an associated IndexState
        instance with its own RWLock for thread-safe operations.
        
        Args:
            library_id: UUID of the library
            
        Returns:
            IndexState for the library
        """
        # Check if state already exists (common case)
        with self._states_lock.read_lock():
            if library_id in self._index_states:
                return self._index_states[library_id]
        
        # Create new state (rare case)
        with self._states_lock.write_lock():
            # Double-check pattern to avoid race conditions
            if library_id not in self._index_states:
                algorithm = IndexAlgo(settings.default_index_type)
                self._index_states[library_id] = IndexState(
                    library_id=library_id,
                    algorithm=algorithm,
                )
                logger.debug(f"Created new index state for library {library_id}")
            
            return self._index_states[library_id]

    def _load_vectors_and_ids(
        self, 
        library_id: UUID, 
        state: IndexState
    ) -> tuple[list[list[float]], list[UUID]]:
        """Load vectors and chunk IDs for index building.
        
        This helper method loads all chunks for a library, ensures they have
        embeddings, validates dimension consistency, and returns the vectors
        and corresponding chunk IDs ready for index building.
        
        Args:
            library_id: UUID of the library to load vectors for
            state: Index state (for dimension tracking)
            
        Returns:
            Tuple of (vectors, chunk_ids) where vectors and chunk_ids have same length
            
        Raises:
            EmbeddingDimensionMismatchError: If embeddings have inconsistent dimensions
        """
        # Load all chunks for this library
        chunks = self._chunk_repo.list_by_library(library_id)
        
        if not chunks:
            return [], []
        
        logger.debug(f"Loading vectors for {len(chunks)} chunks in library {library_id}")
        
        # Ensure all chunks have embeddings
        chunks_with_embeddings = self._ensure_embeddings(chunks)
        
        # Validate dimensions and collect vectors
        vectors = []
        chunk_ids = []
        expected_dim = state.embedding_dim
        
        for chunk in chunks_with_embeddings:
            if not chunk.has_embedding:
                logger.warning(f"Chunk {chunk.id} still missing embedding after generation")
                continue
            
            embedding_dim = chunk.embedding_dim
            
            # Set expected dimension from first valid embedding
            if expected_dim is None:
                expected_dim = embedding_dim
                state.embedding_dim = expected_dim
                logger.debug(f"Set embedding dimension to {expected_dim} for library {library_id}")
            
            # Validate dimension consistency
            if embedding_dim != expected_dim:
                raise EmbeddingDimensionMismatchError(expected_dim, embedding_dim)
            
            vectors.append(chunk.embedding)
            chunk_ids.append(chunk.id)
        
        logger.debug(f"Loaded {len(vectors)} vectors with dimension {expected_dim}")
        return vectors, chunk_ids

    def _ensure_embeddings(self, chunks: list[Chunk]) -> list[Chunk]:
        """Ensure all chunks have embeddings, generating them if necessary.
        
        This helper method processes a list of chunks and ensures each one
        has an embedding vector. It uses batch processing when possible to
        efficiently generate embeddings for multiple chunks.
        
        Args:
            chunks: List of chunks to process
            
        Returns:
            List of chunks with embeddings (may be updated versions)
        """
        # Separate chunks with and without embeddings
        chunks_with_embeddings = []
        chunks_needing_embeddings = []
        
        for chunk in chunks:
            if chunk.has_embedding:
                chunks_with_embeddings.append(chunk)
            else:
                chunks_needing_embeddings.append(chunk)
        
        if not chunks_needing_embeddings:
            return chunks_with_embeddings
        
        logger.info(f"Generating embeddings for {len(chunks_needing_embeddings)} chunks")
        
        try:
            # Extract texts for batch embedding
            texts = [chunk.text for chunk in chunks_needing_embeddings]
            
            # Generate embeddings in batch
            embeddings = self._embedding_client.embed_texts(texts)
            
            # Update chunks with new embeddings
            updated_chunks = []
            for chunk, embedding in zip(chunks_needing_embeddings, embeddings):
                updated_chunk = chunk.update(embedding=embedding)
                # Update in repository
                self._chunk_repo.update(updated_chunk)
                updated_chunks.append(updated_chunk)
            
            logger.info(f"Generated {len(embeddings)} embeddings successfully")
            
            # Return all chunks with embeddings
            return chunks_with_embeddings + updated_chunks
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise IndexBuildError("embedding_generation", str(e)) from e
