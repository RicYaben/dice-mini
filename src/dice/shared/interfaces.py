from collections.abc import Generator
from typing import Protocol, TypeVar

from sqlalchemy import Connection
from sqlmodel import Session, SQLModel

from dice.shared.result import SearchResult

T = TypeVar("T", bound=SQLModel)


class Repository(Protocol):
    def query(self, q: str, bsize: int = 50_000) -> Generator[dict]: ...
    def search(self, q: str) -> SearchResult: ...
    def insert(self, items: list[T], policy=None, con: Connection | None = None): ...
    def connect(self) -> Connection: ...
    def session(self) -> Session: ...
