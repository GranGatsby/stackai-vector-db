"""Library service for business operations."""

from uuid import UUID

from app.domain import Library, LibraryMetadata, LibraryNotFoundError
from app.repositories.ports import (
    ChunkRepository,
    DocumentRepository,
    LibraryRepository,
)


class LibraryService:
    """Service for library business operations."""

    def __init__(
        self,
        repository: LibraryRepository,
        document_repository: DocumentRepository | None = None,
        chunk_repository: ChunkRepository | None = None,
    ) -> None:
        self._repository = repository
        self._document_repository = document_repository
        self._chunk_repository = chunk_repository

    def list_libraries(
        self, limit: int | None = None, offset: int = 0
    ) -> tuple[list[Library], int]:
        libraries = self._repository.list_all(limit=limit, offset=offset)
        total = self._repository.count_all()
        return libraries, total

    def get_library(self, library_id: UUID) -> Library:
        library = self._repository.get_by_id(library_id)
        if not library:
            raise LibraryNotFoundError(str(library_id))
        return library

    def get_library_by_name(self, name: str) -> Library:
        library = self._repository.get_by_name(name)
        if not library:
            raise LibraryNotFoundError(f"Library with name '{name}' not found")
        return library

    def create_library(
        self,
        name: str,
        description: str = "",
        metadata: LibraryMetadata | None = None,
    ) -> Library:
        library = Library.create(name=name, description=description, metadata=metadata)
        return self._repository.create(library)

    def update_library(
        self,
        library_id: UUID,
        name: str | None = None,
        description: str | None = None,
        metadata: LibraryMetadata | None = None,
    ) -> Library:
        existing_library = self.get_library(library_id)
        updated_library = existing_library.update(
            name=name, description=description, metadata=metadata
        )
        return self._repository.update(updated_library)

    def delete_library(self, library_id: UUID) -> bool:
        """Delete library with cascading deletes: chunks -> documents -> library."""
        if not self._repository.exists(library_id):
            return False

        if self._chunk_repository:
            self._chunk_repository.delete_by_library(library_id)
        if self._document_repository:
            self._document_repository.delete_by_library(library_id)

        return self._repository.delete(library_id)

    def library_exists(self, library_id: UUID) -> bool:
        return self._repository.exists(library_id)

    def library_name_exists(self, name: str) -> bool:
        return self._repository.get_by_name(name) is not None
