import logging
import warnings
from collections.abc import Generator, Sequence
from typing import overload
from uuid import uuid4

import pandas as pd
from sqlalchemy import Connection, Select
from sqlmodel import Session, text

from dice.internal.database import Connector, insert_or_ignore
from dice.internal.monitor.health import HealthMonitor
from dice.shared.interfaces import Repository as R
from dice.shared.interfaces import T
from dice.shared.result import SearchResult

from .helpers import normalize_data

warnings.simplefilter(action="ignore", category=UserWarning)

logger = logging.getLogger(__name__)


class Repository(R):
    def __init__(
        self,
        con: Connector,
        bsize: int | None = None,
    ) -> None:
        self.con = con
        self.bsize = bsize

    def load(self, monitor: HealthMonitor) -> "Repository":
        self.monitor = monitor
        monitor.initialize()
        return self

    def connect(self) -> Connection:
        return self.con.connection()

    def session(self) -> Session:
        return self.con.session()

    def insert(
        self, items: list[T], policy=insert_or_ignore, con: Connection | None = None
    ):
        if not items:
            return

        model = type(items[0])
        if not con:
            con = self.connect()

        with Session(con) as s:
            policy(s, model, items)
            s.flush()

    def query(
        self, q: str, bsize: int | None = None, limit: int | None = None
    ) -> Generator[dict, None, None]:
        if limit:
            q = f"{q} LIMIT {limit}"

        with self.connect() as c:
            res = c.execute(text(q))
            cols = [c[0] for c in res.cursor.description]  # type: ignore

            while rows := res.fetchmany(bsize):
                for r in rows:
                    yield dict(zip(cols, r))
                continue

    def querys(self, q: str) -> Generator[dict]:
        for batch in self.query(q):
            yield from batch

    def queryb(
        self,
        q: str,
        bsize: int | None = None,
        norm=normalize_data,
        limit: int | None = None,
    ) -> Generator[pd.DataFrame, None, None]:
        with self.connect() as con:
            norm = norm if norm else lambda x: x
            for batch in query_batch(q, con, bsize, limit):
                df = pd.DataFrame.from_records(batch)
                yield norm(df)

    def queryc(
        self,
        q: str,
        bsize: int | None = None,
        norm=normalize_data,
        limit: int | None = None,
    ) -> tuple[int, Generator[pd.DataFrame, None, None]]:
        with self.connect() as con:
            d = query_count(q, con, limit)
        gen = self.queryb(q, bsize, norm, limit)
        return (d, gen)

    @overload
    def search(
        self, q: str, limit: int | None = None, offset: int | None = None
    ) -> SearchResult: ...

    @overload
    def search(
        self, q: Select, limit: int | None = None, offset: int | None = None
    ) -> SearchResult: ...

    def search(
        self,
        q: str | Select,
        limit: int | None = None,
        offset: int | None = None,
    ) -> SearchResult:
        view = f"tmp_{uuid4().hex}"

        if isinstance(q, Select):
            if limit is not None:
                q = q.limit(limit)

            if offset is not None:
                q = q.offset(offset)

            con = self.connect()

            sql = str(
                q.compile(
                    con,
                    compile_kwargs={"literal_binds": True},
                )
            )
        else:
            if limit is not None:
                q += f" LIMIT {limit}"

            if offset is not None:
                q += f" OFFSET {offset}"

            sql = q
            con = self.connect()

        con.execute(text(f"CREATE TEMP VIEW {view} AS {sql}"))
        return SearchResult(con, view, self.bsize)

    def synchronize(self) -> None:
        return self.monitor.sanity()


def query_batch(
    q: str, con: Connection, bsize: int | None = None, limit: int | None = None
) -> Generator[Sequence, None, None]:
    if limit is not None:
        q = f"""
        SELECT *
        FROM ({q}) AS subq
        LIMIT {int(limit)}
        """

    res = con.execute(text(q)).mappings()
    while rows := res.fetchmany(bsize):
        yield rows


def query_count(q: str, con: Connection, limit: int | None = None) -> int:
    if limit is not None:
        q = f"""
        SELECT *
        FROM ({q}) AS subq
        LIMIT {int(limit)}
        """

    dq = f"WITH ct AS ({q}) SELECT COUNT(*) AS rows FROM ct;"
    d = res[0] if (res := con.execute(text(dq)).fetchone()) else 0
    return d


def new_repository(connector: Connector) -> Repository:
    return Repository(
        con=connector,
    )
