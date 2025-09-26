"""Thread-safe in-memory LibraryRepository implementation."""

from uuid import UUID

from app.domain import Library, LibraryAlreadyExistsError, LibraryNotFoundError
from app.repositories.ports import LibraryRepository
from app.utils import RWLock


class InMemoryLibraryRepository(LibraryRepository):
    """Thread-safe in-memory library storage."""

    def __init__(self) -> None:
        self._libraries: dict[UUID, Library] = {}
        self._name_index: dict[str, UUID] = {}
        self._lock = RWLock()

    def _normalize_name(self, name: str) -> str:
        return name.casefold()

    def list_all(self, limit: int | None = None, offset: int = 0) -> list[Library]:
        with self._lock.read_lock():
            libraries = sorted(self._libraries.values(), key=lambda lib: lib.name.casefold())
            return libraries[offset:offset + limit if limit else None]

    def count_all(self) -> int:
        with self._lock.read_lock():
            return len(self._libraries)

    def get_by_id(self, library_id: UUID) -> Library | None:
        with self._lock.read_lock():
            return self._libraries.get(library_id)

    def get_by_name(self, name: str) -> Library | None:
        with self._lock.read_lock():
            library_id = self._name_index.get(self._normalize_name(name))
            return self._libraries.get(library_id) if library_id else None

    def create(self, library: Library) -> Library:
        with self._lock.write_lock():
            normalized_name = self._normalize_name(library.name)

            if normalized_name in self._name_index:
                raise LibraryAlreadyExistsError(library.name)
            if library.id in self._libraries:
                raise ValueError(f"Library with ID {library.id} already exists")

            self._libraries[library.id] = library
            self._name_index[normalized_name] = library.id
            return library

    def update(self, library: Library) -> Library:
        with self._lock.write_lock():
            if library.id not in self._libraries:
                raise LibraryNotFoundError(str(library.id))

            old_library = self._libraries[library.id]
            old_normalized_name = self._normalize_name(old_library.name)
            new_normalized_name = self._normalize_name(library.name)

            if new_normalized_name != old_normalized_name:
                existing_library_id = self._name_index.get(new_normalized_name)
                if existing_library_id and existing_library_id != library.id:
                    raise LibraryAlreadyExistsError(library.name)

                del self._name_index[old_normalized_name]
                self._name_index[new_normalized_name] = library.id

            self._libraries[library.id] = library
            return library

    def delete(self, library_id: UUID) -> bool:
        with self._lock.write_lock():
            library = self._libraries.get(library_id)
            if not library:
                return False

            del self._libraries[library_id]
            del self._name_index[self._normalize_name(library.name)]
            return True

    def exists(self, library_id: UUID) -> bool:
        with self._lock.read_lock():
            return library_id in self._libraries

    def clear(self) -> None:
        with self._lock.write_lock():
            self._libraries.clear()
            self._name_index.clear()
