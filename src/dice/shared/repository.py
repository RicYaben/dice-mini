from collections.abc import Generator
from typing import TypeVar

from sqlalchemy import Connection
from sqlmodel import Session, select

from .interfaces import Repository
from .models import DatabaseModel, Fingerprint, FingerprintLabel, HostTag, Label, Tag
from .result import SearchResult
from .tools import new_fingerprint, new_host_tag

T = TypeVar("T", bound=DatabaseModel)


class Cache:
    def __init__(self) -> None:
        self.cache: dict[type[DatabaseModel], list[DatabaseModel]] = {}

    # NOTE: May be able to remove the generic here
    def add(self, item: T) -> None:
        self.cache.setdefault(type(item), []).append(item)

    def find(self, query: T) -> T | None:
        for item in self.cache.get(type(query), []):
            if self._matches(item, query):
                return item

        return None

    @staticmethod
    def _matches(item: T, query: T) -> bool:
        for field in query.model_fields:
            value = getattr(query, field)

            if value is not None and getattr(item, field) != value:
                return False

        return True

    def clear(self) -> None:
        self.cache.clear()

    def len(self) -> int:
        return sum(len(items) for items in self.cache.values())


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


class BaseRepo:
    def __init__(self, repo: Repository, name: str) -> None:
        self.cache: Cache = Cache()
        self.csize = 1_000
        self.repo = repo
        self.name = name

    def store(self, *item: DatabaseModel) -> None:
        for i in item:
            self.cache.add(i)
        if self.cache.len() >= self.csize:
            self.flush()

    def flush(self) -> None:
        for vals in self.cache.cache.values():
            self.repo.insert(vals)
        self.cache.clear()

    def query(self, q: str) -> Generator[dict]:
        return self.repo.query(q)

    def search(self, q: str) -> SearchResult:
        return self.repo.search(q)


class FRepo(BaseRepo):
    def fingerprint(self, host: str, record: int, data: dict, protocol: str = "-"):
        fp = fingerprint(self.name, host, record, data, protocol)
        self.store(fp)


class CRepo(BaseRepo):
    def label(self, fp: int, lab: str) -> None:
        with self.repo.connect() as con:
            lb = label(con, fp, lab, cache=self.cache)
            self.store(lb)


class TRepo(BaseRepo):
    def tag(self, host: str, name: str, comment: str | None = None):
        with self.repo.connect() as con:
            t = tag(con, host, name, cache=self.cache, comment=comment)
            self.store(t)
