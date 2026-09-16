from collections.abc import Callable, Generator
from typing import TypeVar

from sqlalchemy import Connection
from sqlmodel import Session, select

from .interfaces import Repository
from .models import DatabaseModel, Fingerprint, FingerprintLabel, HostTag, Label, Tag
from .result import SearchResult
from .tools import new_fingerprint, new_host_tag

T = TypeVar("T", bound=DatabaseModel)


class Cache[T: DatabaseModel]:
    def __init__(self, csize: int, flush_fn: Callable[[list[T]], None]) -> None:
        self.csize = csize
        self.flush_cb = flush_fn
        self.cache: list[T] = []

    def add(self, *items: T) -> None:
        for item in items:
            self.cache.append(item)
            if len(self.cache) >= self.csize:
                self.flush()

    def find(self, query: T) -> T | None:
        for item in self.cache:
            if self._matches(item, query):
                return item

        return None

    @staticmethod
    def _matches(item: T, query: T) -> bool:
        for field, value in type(query).model_fields.items():
            if value is not None and getattr(item, field) != value:
                return False

        return True

    def clear(self) -> None:
        self.cache.clear()

    def flush(self) -> None:
        self.flush_cb(self.cache)
        self.cache.clear()


def label(con: Connection, fp: int, lab: str, cache: Cache) -> FingerprintLabel:
    if lb := cache.find(Label(name=lab)):
        return FingerprintLabel(fingerprint_id=fp, label_id=lb.id)

    with Session(con) as s:
        q = select(Label).where(Label.name == lab)
        slab = s.exec(q).one()
        cache.add(slab)
        return FingerprintLabel(fingerprint_id=fp, label_id=slab.id)


def tag(
    con: Connection,
    host: str,
    tag: str,
    cache: Cache,
    comment: str | None,
) -> HostTag:
    if t := cache.find(Tag(name=tag)):
        assert t.id is not None
        return new_host_tag(host, t.id, comment)

    with Session(con) as s:
        q = select(Tag).where(Tag.name == tag)
        t = s.exec(q).one()
        cache.add(t)

        assert t.id is not None
        return new_host_tag(host, t.id, comment)


def fingerprint(
    mod: str, host: str, record: int, data: dict, protocol: str
) -> Fingerprint:
    return new_fingerprint(mod, host, record, data, protocol)


class BaseRepo[T: DatabaseModel]:
    def __init__(self, repo: Repository, name: str) -> None:
        self.cache: Cache[T] = Cache(1_000, self.repo.insert)
        self.repo = repo
        self.name = name

    def store(self, *item: T) -> None:
        self.cache.add(*item)

    def flush(self) -> None:
        self.cache.flush()

    def query(self, q: str) -> Generator[dict]:
        return self.repo.query(q)

    def search(self, q: str) -> SearchResult:
        return self.repo.search(q)


class FRepo(BaseRepo[Fingerprint]):
    def fingerprint(self, host: str, record: int, data: dict, protocol: str = "-"):
        fp = fingerprint(self.name, host, record, data, protocol)
        self.store(fp)


class CRepo(BaseRepo[FingerprintLabel]):
    def label(self, fp: int, lab: str) -> None:
        with self.repo.connect() as con:
            lb = label(con, fp, lab, cache=self.cache)
            self.store(lb)


class TRepo(BaseRepo[HostTag]):
    def tag(self, host: str, name: str, comment: str | None = None):
        with self.repo.connect() as con:
            t = tag(con, host, name, cache=self.cache, comment=comment)
            self.store(t)
