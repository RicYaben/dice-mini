from collections.abc import Generator
from typing import Protocol, TypeVar

from sqlalchemy import Connection, Select
from sqlmodel import Session, SQLModel

from dice.shared.result import SearchResult

T = TypeVar("T", bound=SQLModel)


class Repository(Protocol):
    def query(self, q: str, bsize: int = 50_000) -> Generator[dict]: ...
    def search(
        self, q: Select | str, limit: int | None = None, offset: int | None = None
    ) -> SearchResult: ...
    def insert(self, items: list[T], policy=None, con: Connection | None = None): ...
    def connect(self) -> Connection: ...
    def session(self) -> Session: ...
    def synchronize(self) -> None: ...  # TODO: return a health report?
