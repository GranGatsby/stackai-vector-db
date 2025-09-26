"""Index service for vector index management."""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID

from app.clients import EmbeddingClient, create_embedding_client
from app.core.config import settings
from app.domain import (
    Chunk,
    ChunkMetadata,
    EmbeddingDimensionMismatchError,
    LibraryNotFoundError,
    VectorIndexBuildError,
    VectorIndexNotBuiltError,
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
    """Immutable snapshot of a vector index for thread-safe access."""

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
    """Mutable state for a library's vector index."""

    library_id: UUID
    algorithm: IndexAlgo = IndexAlgo.LINEAR
    is_dirty: bool = True
    dirty_count: int = 0
    total_chunks: int = 0
    current_snapshot: IndexSnapshot | None = None
    lock: RWLock = field(default_factory=RWLock)
    embedding_dim: int | None = None

    @property
    def is_built(self) -> bool:
        """Check if this index has been built."""
        return self.current_snapshot is not None and self.current_snapshot.is_built

    @property
    def version(self) -> int:
        """Get the current version number."""
        return self.current_snapshot.version if self.current_snapshot else 0

    @property
    def built_at(self) -> float | None:
        """Get the timestamp when this index was last built."""
        return self.current_snapshot.built_at if self.current_snapshot else None

    @property
    def size(self) -> int:
        """Get the number of vectors in the current index."""
        return self.current_snapshot.size if self.current_snapshot else 0

    def should_rebuild(self, rebuild_threshold: float) -> bool:
        """Determine if the index should be rebuilt based on dirty ratio."""
        if not self.is_built:
            return True

        if self.total_chunks == 0:
            return True

        dirty_ratio = self.dirty_count / max(self.total_chunks, 1)
        return dirty_ratio >= rebuild_threshold


@dataclass(frozen=True)
class IndexStatus:
    """Status information for a library's vector index."""

    library_id: UUID
    algorithm: IndexAlgo
    is_built: bool
    is_dirty: bool
    size: int
    embedding_dim: int | None
    built_at: float | None
    version: int
    dirty_count: int
    build_duration: float | None = None


class IndexService:
    """Service for vector index management per library."""

    def __init__(
        self,
        library_repository: LibraryRepository,
        chunk_repository: ChunkRepository,
        embedding_client: EmbeddingClient | None = None,
        rebuild_threshold: float | None = None,
    ) -> None:
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
        """Get the current status of a library's index."""
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
        """Mark a library's index as dirty (needing rebuild)"""
        # Verify library exists
        if not self._library_repo.exists(library_id):
            raise LibraryNotFoundError(str(library_id))

        # Get or create index state
        state = self._get_or_create_state(library_id)

        # Use write lock for state modification
        with state.lock.write_lock():
            was_dirty = state.is_dirty
            state.is_dirty = True
            state.dirty_count += 1

            # Calculate dirty ratio for rebuild decision
            dirty_ratio = (
                state.dirty_count / max(state.total_chunks, 1)
                if state.total_chunks > 0
                else 0
            )
            should_rebuild = state.should_rebuild(self._rebuild_threshold)

        if not was_dirty:
            logger.info(
                f"Index for library {library_id} marked as dirty "
                f"(dirty_count={state.dirty_count}, total_chunks={state.total_chunks}, "
                f"dirty_ratio={dirty_ratio:.2%}, should_rebuild={should_rebuild})"
            )
        else:
            logger.debug(
                f"Index for library {library_id} dirty count incremented "
                f"(dirty_count={state.dirty_count}, dirty_ratio={dirty_ratio:.2%})"
            )

    def build(
        self, library_id: UUID, algorithm: IndexAlgo | str | None = None
    ) -> IndexStatus:
        """Build or rebuild the vector index for a library."""
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

                # Track build duration
                build_start_time = time.time()
                build_duration = None

                if not vectors:
                    logger.info(
                        f"No chunks found for library {library_id}, creating empty index"
                    )
                    # Set default embedding dimension for empty library
                    default_dim = settings.default_embedding_dim
                    if state.embedding_dim is None:
                        state.embedding_dim = default_dim

                    # Create empty index snapshot
                    empty_index = create_index(
                        index_type=state.algorithm.value, dimension=state.embedding_dim
                    )

                    build_duration = time.time() - build_start_time
                    new_snapshot = IndexSnapshot(
                        index=empty_index,
                        chunk_ids=[],
                        built_at=time.time(),
                        version=(state.version + 1),
                        embedding_dim=state.embedding_dim,
                    )
                else:
                    # Build index with vectors
                    index = create_index(
                        index_type=state.algorithm.value, dimension=state.embedding_dim
                    )

                    # Build the index
                    index.build(vectors)

                    build_duration = time.time() - build_start_time
                    logger.info(
                        f"Built {state.algorithm.value} index for library {library_id} "
                        f"with {len(vectors)} vectors in {build_duration:.3f}s"
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
                    f"(version={new_snapshot.version}, size={new_snapshot.size}, duration={build_duration:.3f}s)"
                )

                # Return status with build duration
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
                    build_duration=build_duration,
                )

            except Exception as e:
                error_msg = f"Failed to build index for library {library_id}: {e}"
                logger.error(error_msg)
                raise VectorIndexBuildError(str(library_id), str(e)) from e

    def query(
        self, library_id: UUID, query_vector: list[float], k: int = 10
    ) -> list[tuple[UUID, float]]:
        """Execute a k-NN query on a library's index"""
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
                logger.warning(
                    f"Query failed for library {library_id}: index not built"
                )
                raise VectorIndexNotBuiltError(str(library_id))

            # Validate embedding dimension
            if len(query_vector) != snapshot.embedding_dim:
                logger.error(
                    f"Query failed for library {library_id}: dimension mismatch "
                    f"(expected {snapshot.embedding_dim}, got {len(query_vector)})"
                )
                raise EmbeddingDimensionMismatchError(
                    snapshot.embedding_dim, len(query_vector)
                )

            # Log query details
            logger.debug(
                f"Executing query on library {library_id}: "
                f"algorithm={state.algorithm.value}, k={k}, "
                f"index_size={snapshot.size}, embedding_dim={snapshot.embedding_dim}"
            )

            # Execute k-NN query on the index
            query_start_time = time.time()
            query_results = snapshot.index.query(query_vector, k)
            query_duration = time.time() - query_start_time

            # Convert vector indices to chunk IDs
            chunk_results = []
            for vector_index, distance in query_results:
                if vector_index < len(snapshot.chunk_ids):
                    chunk_id = snapshot.chunk_ids[vector_index]
                    chunk_results.append((chunk_id, distance))

            logger.info(
                f"Query completed for library {library_id}: "
                f"found {len(chunk_results)} results in {query_duration * 1000:.2f}ms "
                f"using {state.algorithm.value} algorithm"
            )

            return chunk_results

    def _get_or_create_state(self, library_id: UUID) -> IndexState:
        """Get or create index state for a library."""
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
        self, library_id: UUID, state: IndexState
    ) -> tuple[list[list[float]], list[UUID]]:
        """Load vectors and chunk IDs for index building."""
        # Load all chunks for this library
        chunks = self._chunk_repo.list_by_library(library_id)

        if not chunks:
            return [], []

        logger.debug(
            f"Loading vectors for {len(chunks)} chunks in library {library_id}"
        )

        # Ensure all chunks have embeddings
        chunks_with_embeddings = self._ensure_embeddings(chunks)

        # Validate dimensions and collect vectors
        vectors = []
        chunk_ids = []
        expected_dim = state.embedding_dim

        for chunk in chunks_with_embeddings:
            if not chunk.has_embedding:
                logger.warning(
                    f"Chunk {chunk.id} still missing embedding after generation"
                )
                continue

            embedding_dim = chunk.embedding_dim

            # Set expected dimension from first valid embedding
            if expected_dim is None:
                expected_dim = embedding_dim
                state.embedding_dim = expected_dim
                logger.debug(
                    f"Set embedding dimension to {expected_dim} for library {library_id}"
                )

            # Validate dimension consistency
            if embedding_dim != expected_dim:
                raise EmbeddingDimensionMismatchError(expected_dim, embedding_dim)

            vectors.append(chunk.embedding)
            chunk_ids.append(chunk.id)

        logger.debug(f"Loaded {len(vectors)} vectors with dimension {expected_dim}")
        return vectors, chunk_ids

    def _ensure_embeddings(self, chunks: list[Chunk]) -> list[Chunk]:
        """Ensure all chunks have embeddings, generating them if necessary."""
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

        logger.info(
            f"Generating embeddings for {len(chunks_needing_embeddings)} chunks"
        )

        try:
            # Extract texts for batch embedding
            texts = [chunk.text for chunk in chunks_needing_embeddings]

            # Generate embeddings in batch
            embedding_result = self._embedding_client.embed_texts(texts)

            # Update chunks with new embeddings and metadata
            updated_chunks = []
            for chunk, embedding in zip(
                chunks_needing_embeddings, embedding_result.embeddings, strict=False
            ):
                # Update metadata with embedding information
                current_metadata = chunk.metadata or ChunkMetadata()
                updated_metadata = ChunkMetadata(
                    # Preserve existing metadata
                    chunk_type=current_metadata.chunk_type,
                    section=current_metadata.section,
                    page_number=current_metadata.page_number,
                    confidence=current_metadata.confidence,
                    language=current_metadata.language,
                    tags=current_metadata.tags,
                    similarity_threshold=current_metadata.similarity_threshold,
                    processed_at=current_metadata.processed_at,
                    # Update embedding metadata
                    embedding_model=embedding_result.model_name,
                    embedding_dim=embedding_result.embedding_dim,
                )

                updated_chunk = chunk.update(
                    embedding=embedding, metadata=updated_metadata
                )
                # Update in repository
                self._chunk_repo.update(updated_chunk)
                updated_chunks.append(updated_chunk)

            logger.info(
                f"Generated {len(embedding_result.embeddings)} embeddings successfully"
            )

            # Return all chunks with embeddings
            return chunks_with_embeddings + updated_chunks

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise VectorIndexBuildError("embedding_generation", str(e)) from e
