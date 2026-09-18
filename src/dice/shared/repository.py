from collections.abc import Callable, Generator
from typing import TypeVar

from sqlalchemy import Connection
from sqlmodel import Session, select

from .interfaces import Repository
from .models import Fingerprint, FingerprintLabel, HostTag, Label, ResultsModel, Tag
from .result import SearchResult
from .tools import new_fingerprint, new_host_tag

T = TypeVar("T", bound=ResultsModel)


class Cache[T: ResultsModel]:
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
        for field, value in query.model_dump(exclude_unset=True).items():
            if getattr(item, field) != value:
                return False

        return True

    def clear(self) -> None:
        self.cache.clear()

    def flush(self) -> None:
        if not self.cache:
            return

        items = self.cache
        self.flush_cb(items)
        self.clear()


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


class BaseRepo[T: ResultsModel]:
    def __init__(self, repo: Repository, name: str) -> None:
        self.repo = repo
        self.name = name
        self.cache = Cache[T](1_000, repo.insert)
        self.caches: list[Cache] = [self.cache]

    def store(self, *items: T) -> None:
        self.cache.add(*items)

    def flush(self) -> None:
        for cache in self.caches:
            cache.flush()

    def query(self, q: str) -> Generator[dict]:
        return self.repo.query(q)

    def search(self, q: str) -> SearchResult:
        return self.repo.search(q)


class FRepo(BaseRepo[Fingerprint]):
    def fingerprint(self, host: str, record: int, data: dict, protocol: str = "-"):
        fp = fingerprint(self.name, host, record, data, protocol)
        self.store(fp)


class CRepo(BaseRepo[FingerprintLabel]):
    def __init__(self, repo: Repository, name: str) -> None:
        super().__init__(repo, name)

        self.label_cache = Cache[Label](100, repo.insert)
        self.caches.append(self.label_cache)

    def label(self, fp: int, lab: str) -> None:
        with self.repo.connect() as con:
            lb = label(con, fp, lab, cache=self.label_cache)

        self.store(lb)


class TRepo(BaseRepo[HostTag]):
    def __init__(self, repo: Repository, name: str) -> None:
        super().__init__(repo, name)

        self.tag_cache = Cache[Tag](100, repo.insert)
        self.caches.append(self.tag_cache)

    def tag(
        self,
        host: str,
        name: str,
        comment: str | None = None,
    ) -> None:
        with self.repo.connect() as con:
            t = tag(
                con,
                host,
                name,
                cache=self.tag_cache,
                comment=comment,
            )

        self.store(t)
