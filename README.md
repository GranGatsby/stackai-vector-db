# StackAI Vector Database

A REST API for indexing and querying documents in a Vector Database, developed as part of the StackAI technical interview process.

## Features

- **Complete REST API** for CRUD operations on libraries, documents, and chunks
- **k-NN vector search** with multiple indexing algorithms
- **Structured logging** with request tracking and multiple formatters
- **Request middleware** with unique IDs and timing metrics
- **Docker containerization**
- **Integrated development tools** (Black, Ruff, Pre-commit)
- **Comprehensive test suite** with coverage of main components
- **Thread-safe operations** with read-write locks
- **Automatic index selection** based on data characteristics
- **Flexible embedding system** with Cohere API integration and fallback

## Requirements

- Python 3.12+
- Docker (optional)
- Cohere API key for embeddings (optional - uses fake client if not provided)

## Installation & Setup

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd stackai-vector-db
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate  # Windows
   ```

3. **Install development dependencies**
   ```bash
   make setup
   # or manually:
   pip install -e ".[dev]"
   pre-commit install
   ```

4. **Configure environment variables**
   ```bash
   cp env.example .env
   # Edit .env with your Cohere API key (optional)
   ```

### Docker

```bash
# Build image
make docker-build

# Run container
make docker-run
```

## 🏃‍♂️ Usage

### Running the Application

```bash
# Development mode
make run

# Or directly
python -m app.main
```

The API will be available at `http://localhost:8000`

### API Documentation

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Health Check

```bash
curl http://localhost:8000/api/v1/health
```

### Basic API Usage Example

```bash
# 1. Create a library
curl -X POST "http://localhost:8000/api/v1/libraries" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Documents",
    "description": "Collection of technical documents",
    "metadata": {"category": "technical", "is_public": true}
  }'

# 2. Create a document in the library
curl -X POST "http://localhost:8000/api/v1/libraries/{library_id}/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Vector Database Guide",
    "content": "This is a comprehensive guide about vector databases...",
    "metadata": {"author": "John Doe", "format": "markdown"}
  }'

# 3. Create chunks in the document (with automatic embedding)
curl -X POST "http://localhost:8000/api/v1/documents/{document_id}/chunks" \
  -H "Content-Type: application/json" \
  -d '{
    "chunks": [
      {
        "text": "Vector databases are specialized databases for storing and querying high-dimensional vectors.",
        "start_index": 0,
        "end_index": 95
      }
    ],
    "compute_embedding": true
  }'

# 4. Build the vector index
curl -X POST "http://localhost:8000/api/v1/libraries/{library_id}/index" \
  -H "Content-Type: application/json" \
  -d '{"algorithm": "ivf"}'

# 5. Search by text
curl -X POST "http://localhost:8000/api/v1/libraries/{library_id}/query/text" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "What are vector databases?",
    "k": 5
  }'
```

## Testing

```bash
# Run all tests
make test

# Run specific test files
pytest tests/test_health.py -v
pytest tests/test_indexes.py -v
pytest tests/test_search_service.py -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html

# Run only unit tests
pytest tests/ -m "not integration" -v
```

### Test Coverage

The project includes comprehensive tests covering:

- **Unit tests**: Domain entities, services, repositories, and indexing algorithms
- **Integration tests**: API endpoints, database operations, and search functionality
- **Concurrency tests**: Thread-safety of indexing operations
- **Edge case tests**: Error handling, validation, and boundary conditions

Key test files:
- `test_indexes.py`: Vector indexing algorithms (Linear, KDTree, IVF)
- `test_search_service.py`: Search functionality and embedding integration
- `test_concurrency.py`: Thread-safety and concurrent operations
- `test_*_api.py`: REST API endpoints and validation
- `test_*_service.py`: Business logic and use cases

## Development Tools

```bash
# Format code
make format

# Linting
make lint

# Type checking
make type-check

# Run pre-commit hooks
make pre-commit

# All checks
make check
```

## Architecture & Design

The project follows **Clean Architecture** principles with **Domain-Driven Design (DDD)**:

```
┌─────────────────────────────────────────────────────────────┐
│                        API Layer                            │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ FastAPI Routers (HTTP endpoints, validation)        │    │
│  │ Error handlers, middleware, dependency injection    │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                     Service Layer                           │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Application services (use cases, business logic)    │    │
│  │ LibraryService, DocumentService, ChunkService       │    │
│  │ IndexService, SearchService                         │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                     Domain Layer                            │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Entities (Library, Document, Chunk)                 │    │
│  │ Domain errors, business rules, invariants           │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                 Infrastructure Layer                        │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Repositories (in-memory implementations)            │    │
│  │ Vector indexes (Linear, KDTree, IVF)                │    │
│  │ External clients (Cohere API, FakeClient)           │    │
│  │ Utils (RWLock, logging, configuration)              │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Patterns

1. **Repository Pattern**: Abstracts data access with protocol-based interfaces
2. **Dependency Injection**: Services receive dependencies through constructors
3. **Factory Pattern**: Index creation based on algorithm type and data characteristics
4. **Strategy Pattern**: Multiple indexing algorithms with common interface
5. **Immutable Snapshots**: Thread-safe read access to index data
6. **Command Pattern**: Structured request/response schemas for API operations

### Project Structure

```
stackai-vector-db/
├── app/
│   ├── api/v1/                 # API layer
│   │   ├── routers/            # FastAPI route handlers
│   │   ├── deps.py             # Dependency injection
│   │   └── errors.py           # Error handling
│   ├── services/               # Application services
│   │   ├── library_service.py  # Library management
│   │   ├── document_service.py # Document management
│   │   ├── chunk_service.py    # Chunk management
│   │   ├── index_service.py    # Vector indexing
│   │   └── search_service.py   # Search operations
│   ├── domain/                 # Domain layer
│   │   ├── entities.py         # Business entities
│   │   └── errors.py           # Domain exceptions
│   ├── repositories/           # Data access layer
│   │   ├── ports.py            # Repository interfaces
│   │   └── in_memory/          # In-memory implementations
│   ├── indexes/                # Vector indexing algorithms
│   │   ├── base.py             # Common interfaces
│   │   ├── linear.py           # Linear scan index
│   │   ├── kdtree.py           # KD-Tree index
│   │   ├── ivf.py              # IVF index
│   │   └── manager.py          # Index management
│   ├── clients/                # External service clients
│   │   └── embedding.py        # Cohere API client
│   ├── schemas/                # Pydantic DTOs
│   ├── core/                   # Configuration & logging
│   ├── utils/                  # Utilities (RWLock, etc.)
│   └── main.py                 # Application entry point
├── tests/                      # Test suite
├── Dockerfile                  # Container image
├── Makefile                    # Development commands
└── pyproject.toml             # Project configuration
```

## Vector Indexing Algorithms

### Implemented Algorithms

#### 1. LinearScanIndex (Baseline)

**Purpose**: Provides exact results and serves as a reference implementation.

- **Time Complexity**: 
  - Build: O(N) - Simply stores vectors in memory
  - Query: O(N × D) - Exhaustive scan through all vectors
  - Add/Remove: O(1) - Direct list operations
- **Space Complexity**: O(N × D)
- **Characteristics**:
  - Exact results (no approximation)
  - No preprocessing required
  - Excellent for small datasets (<1K vectors)
  - Reliable baseline for correctness comparison
  - Supports both Euclidean and Cosine distance metrics

#### 2. KDTreeIndex (Educational & Low-Dimensional Use)

**Purpose**: Demonstrates spatial partitioning techniques, effective for specific low-dimensional scenarios.

- **Time Complexity**:
  - Build: O(N log N) - Recursive partitioning with median finding
  - Query: O(log N) average, O(N) worst case
  - Add/Remove: O(log N) - May require rebalancing
- **Space Complexity**: O(N) - Tree structure overhead
- **Characteristics**:
  - Excellent for low dimensions (D ≤ 20)
  - Performance degrades in high dimensions (curse of dimensionality)
  - Exact results
  - Memory-efficient tree structure
  - **Educational value**: Classic algorithm for understanding spatial data structures

**Note**: While KDTree was included more for educational purposes than practical optimization, it demonstrates important concepts in spatial indexing and performs well in its intended use case (low-dimensional data).

#### 3. IVFIndex (Inverted File - Production-Ready)

**Purpose**: Scalable approximate search for large datasets and high-dimensional embeddings.

- **Time Complexity**:
  - Build: O(N × C × I) - N vectors, C clusters, I k-means iterations
  - Query: O(P × M + k) - P probes, M average vectors per cluster
  - Add: O(C) - Find nearest cluster and add to inverted list
  - Remove: O(M) - Search and remove from inverted list
- **Space Complexity**: O(N + C × D) - Vectors plus cluster centroids
- **Characteristics**:
  - Excellent scalability for large datasets (>10K vectors)
  - Approximate results (tunable via nprobe parameter)
  - Good performance in high dimensions
  - Efficient incremental updates
  - **Production-ready**: Similar to algorithms used in FAISS and other production systems

### Automatic Algorithm Selection

The system intelligently selects the optimal algorithm based on data characteristics:

```python
def recommend_index_type(n_vectors: int, dimension: int, accuracy_priority: bool = True) -> str:
    # Small datasets: linear scan is fine and gives exact results
    if n_vectors < 1000:
        return "linear"
    
    # Low dimensions: KD-Tree works well
    if dimension <= 20 and n_vectors < 50000:
        return "kdtree"
    
    # Large datasets or high dimensions: IVF scales better
    if n_vectors >= 10000 or dimension > 50:
        return "ivf"
    
    # Medium datasets: choose based on accuracy priority
    return "kdtree" if accuracy_priority and dimension <= 20 else "ivf"
```

### Algorithm Selection Rationale

**Why these 3 algorithms?**

1. **LinearScan**: Essential baseline that guarantees exact results and serves as a correctness reference
2. **KDTree**: Classic algorithm demonstrating spatial partitioning techniques, valuable for educational purposes and specific low-dimensional use cases
3. **IVF**: Modern algorithm used in production systems, scalable and practical for real-world applications

This combination covers the complete spectrum: **accuracy vs speed**, **small vs large datasets**, and **low vs high dimensions**.

## Concurrency & Thread Safety

### Design Decisions

#### Read-Write Locks (RWLock)
- **Multiple concurrent readers** allowed for query operations
- **Exclusive writers** for index building and data modifications
- **Per-library locking** to avoid global bottlenecks
- **Context manager support** for safe lock acquisition/release

```python
class RWLock:
    def read_lock(self):
        # Multiple readers can acquire simultaneously
        
    def write_lock(self):
        # Exclusive access for writers
```

#### Immutable Snapshots
- **Copy-on-write semantics** for index updates
- **Atomic snapshot swapping** for consistent reads during rebuilds
- **Version tracking** to detect stale references

```python
@dataclass(frozen=True)
class IndexSnapshot:
    index: VectorIndex
    chunk_ids: list[UUID]
    built_at: float
    version: int
    embedding_dim: int
```

#### Single-Writer Principle
- **One writer per library** at any time
- **Queued write operations** to prevent race conditions
- **Consistent state transitions** during index operations

### Concurrency Benefits
- **High read throughput** with concurrent queries
- **Non-blocking reads** during index rebuilds
- **Data consistency** guaranteed through proper locking
- **Deadlock prevention** through lock ordering

## Embedding System Design

### Hybrid Embedding Strategy

The system implements both **eager** and **lazy** embedding computation:

#### Eager Embedding (Chunk Creation)
```python
# Embedding computed immediately when chunk is created
chunk_service.create_chunks(
    document_id=doc_id,
    chunks_data=chunks,
    compute_embedding=True  # Compute now
)
```

#### Lazy Embedding (Index Building)
```python
# Missing embeddings computed during index building
def _ensure_embeddings(self, chunks: list[Chunk]) -> list[Chunk]:
    chunks_to_embed = [chunk for chunk in chunks if not chunk.has_embedding]
    # Compute missing embeddings automatically
```

### Embedding Client Architecture

- **Cohere API Client**: Production embedding generation
- **Fake Client**: Development/testing with deterministic embeddings
- **Automatic fallback**: Uses fake client when API key not provided
- **Batch processing**: Efficient bulk embedding computation
- **Error handling**: Robust error recovery and retries

### Distance Metrics

Currently uses **Euclidean distance** across all algorithms:
- **Consistent similarity semantics** across different index types
- **Meaningful similarity thresholds** for filtering results
- **Prepared for Cosine distance** in LinearScan (configurable)

## Configuration

All configuration is managed through environment variables using Pydantic Settings:

### Core Settings
- `COHERE_API_KEY`: Cohere API key (optional - uses fake client if not provided)
- `DEFAULT_INDEX_TYPE`: Default index algorithm (linear, kdtree, ivf)
- `MAX_CHUNKS_PER_LIBRARY`: Maximum chunks per library (default: 10,000)
- `DEFAULT_EMBEDDING_DIM`: Default embedding dimension (default: 1024)

### Performance Settings
- `INDEX_REBUILD_THRESHOLD`: Rebuild when X% of data changes (default: 0.1)
- `MAX_KNN_RESULTS`: Maximum k value for search (default: 1000)
- `DEFAULT_KNN_RESULTS`: Default k value (default: 10)

### Logging Configuration
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `LOG_FORMAT_GENERAL`: Format for general logs
- `LOG_FORMAT_REQUEST`: Format for request logs with structured fields

See `env.example` for all available configuration options.

## Known Limitations

### Current Limitations

1. **In-Memory Storage Only**
   - No persistence to disk implemented
   - Data lost on application restart
   - Memory usage grows with dataset size

2. **No Advanced Features**
   - Metadata filtering not implemented
   - No leader-follower architecture
   - No Python SDK client

3. **Scalability Constraints**
   - Not optimized for very large datasets (>1M vectors)
   - Single-node deployment only
   - No horizontal scaling support

4. **Embedding Limitations**
   - Limited to single embedding model per deployment
   - No embedding versioning or migration
   - Fixed distance metric per index type

5. **Missing Production Features**
   - No authentication/authorization
   - No rate limiting
   - No monitoring/metrics collection
   - No backup/restore functionality

### Performance Considerations

- **Memory usage**: Proportional to dataset size
- **Index rebuild cost**: Can be expensive for large datasets
- **Cold start**: No index and memory persistence means rebuilding on restart

## Future Extensions

### Priority Enhancements

#### 1. Improved Embedding Management
- **Embedding status tracking**: `"up_to_date" | "stale" | "missing"`
- **Embedding source tracking**: `"internal" | "external" | "none"`
- **Smart recomputation**: Only recompute when text changes
- **Embedding versioning**: Handle model updates gracefully

#### 2. Enhanced Index Management
- **Chunk indexing status**: Implement `is_indexed` field for user visibility
- **Automatic rebuild triggers**: Rebuild when dirty ratio exceeds threshold
- **Background rebuilding**: Non-blocking index updates
- **Index persistence**: Save/load indexes to/from disk

#### 3. Advanced Search Features
- **Configurable distance metrics**: Choose between Euclidean and Cosine
- **Hybrid search**: Combine vector similarity with keyword search
- **Result ranking**: Custom scoring and ranking algorithms

### Extended Features (From Task Requirements)

#### 1. Metadata Filtering
```python
# Example: Search with metadata filters
search_service.query_text(
    library_id=lib_id,
    text="machine learning",
    k=10,
    filters={
        "metadata.chunk_type": "paragraph",
        "metadata.confidence": {"$gte": 0.8},
        "metadata.tags": {"$in": ["technical", "ai"]}
    }
)
```

#### 2. Disk Persistence
- **Index serialization**: Save/load index structures
- **WAL (Write-Ahead Logging)**: Ensure durability
- **Incremental backups**: Efficient data protection
- **Recovery mechanisms**: Handle corruption gracefully

#### 3. Leader-Follower Architecture
- **Read replicas**: Scale read operations
- **Automatic failover**: High availability
- **Data replication**: Consistent data across nodes
- **Leader election**: Distributed consensus

#### 4. Python SDK Client
```python
from stackai_vector_db import VectorDBClient

client = VectorDBClient("http://localhost:8000")
library = client.create_library("My Library")
document = library.create_document("My Document", content="...")
results = library.search("query text", k=5)
```
---

**Built for StackAI Technical Interview**

### Technical Interview Deliverables

### Answering the Task Requirements

**✅ Define Chunk, Document and Library classes**
- Implemented as immutable domain entities with Pydantic validation
- Fixed metadata schemas for simplified validation
- Factory methods for consistent entity creation

**✅ Implement indexing algorithms**
- LinearScan: O(N×D) query, exact results, baseline implementation
- KDTree: O(log N) average query, educational value, low-dimensional optimization
- IVF: O(P×M+k) query, production-ready scalability for large datasets
- Automatic algorithm selection based on data characteristics

**✅ Ensure no data races**
- Read-Write locks for concurrent access control
- Immutable snapshots for consistent reads during rebuilds
- Per-library locking to avoid global bottlenecks
- Atomic operations for index updates

**✅ CRUD operations with Services**
- Decoupled API endpoints from business logic
- Service layer implementing use cases
- Repository pattern for data access abstraction
- Dependency injection for loose coupling

**✅ API layer implementation**
- FastAPI with automatic OpenAPI documentation
- Structured error handling with consistent responses
- Request validation using Pydantic schemas
- RESTful endpoint design with proper HTTP status codes

**✅ Docker containerization**
- Multi-stage Dockerfile for optimized image size
- Development and production configurations
- Easy deployment with make commands
