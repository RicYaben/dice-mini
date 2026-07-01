from typing import Generator, Optional

from sqlalchemy import Connection
from sqlmodel import select

from .interfaces import Repository
from .models import Fingerprint, FingerprintLabel, HostTag, Label, Model, Tag
from .result import SearchResult
from .tools import new_fingerprint, new_host_tag


def label(
    con: Connection, fp: int, lab: str, cache: list[Label] = []
) -> FingerprintLabel:
    if lb := next(filter(lambda x: x.name == lab, cache)):
        assert lb.id
        return FingerprintLabel(fingerprint_id=fp, label_id=lb.id)

    q = select(Label).where(Label.name == lab)
    slab = con.execute(q).scalar()
    assert slab
    assert slab.id
    cache.append(slab)
    return FingerprintLabel(fingerprint_id=fp, label_id=lb.id)


def tag(
    con: Connection, host: str, tag: str, comment: Optional[str], cache: list[Tag] = []
) -> HostTag:
    if t := next(filter(lambda x: x.name == tag, cache)):
        assert t.id
        return new_host_tag(host, t.id, comment)

    q = select(Tag).where(Tag.name == tag)
    t = con.execute(q).scalar()
    assert t
    assert t.id
    cache.append(t)
    return new_host_tag(host, t.id)


def fingerprint(
    mod: str, host: str, record: int, data: dict, protocol: str
) -> Fingerprint:
    return new_fingerprint(mod, host, record, data, protocol)


class BaseRepo:
    def __init__(self, repo: Repository, name: str) -> None:
        self.cache: list = []
        self.csize = 1_000
        self.repo = repo
        self.name = name

    def store(self, *item: Model) -> None:
        self.cache.extend(item)
        if len(self.cache) >= self.csize:
            self.flush()

    def flush(self) -> None:
        if not self.cache:
            return
        self.repo.insert(self.cache)
        self.cache = []

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
    def tag(self, host: str, name: str, comment: Optional[str] = None):
        with self.repo.connect() as con:
            t = tag(con, host, name, comment, cache=self.cache)
            self.store(t)
