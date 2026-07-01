import logging
import warnings
from typing import Generator, Optional, Sequence
from uuid import uuid4

import pandas as pd
from sqlalchemy import Connection
from sqlmodel import Session, text

from dice.shared.interfaces import Repository as R
from dice.shared.interfaces import T
from dice.shared.result import SearchResult

from .config import DEFAULT_BSIZE
from .database import Connector, insert_or_ignore
from .health import HealthMonitor
from .helpers import normalize_data

warnings.simplefilter(action="ignore", category=UserWarning)

logger = logging.getLogger(__name__)


class Repository(R):
    def __init__(
        self,
        con: Connector,
    ) -> None:
        self.con = con

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
        self, q: str, bsize: int = DEFAULT_BSIZE, limit: Optional[int] = None
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
            for record in batch:
                yield record

    def queryb(
        self,
        q: str,
        bsize: int = DEFAULT_BSIZE,
        norm=normalize_data,
        limit: Optional[int] = None,
    ) -> Generator[pd.DataFrame, None, None]:
        with self.connect() as con:
            norm = norm if norm else lambda x: x
            for batch in query_batch(q, con, bsize, limit):
                df = pd.DataFrame.from_records(batch)
                yield norm(df)

    def queryc(
        self,
        q: str,
        bsize: int = DEFAULT_BSIZE,
        norm=normalize_data,
        limit: Optional[int] = None,
    ) -> tuple[int, Generator[pd.DataFrame, None, None]]:
        with self.connect() as con:
            d = query_count(q, con, limit)
        gen = self.queryb(q, bsize, norm, limit)
        return (d, gen)

    def search(self, q: str, limit: Optional[int] = None) -> SearchResult:
        if limit:
            q += f" LIMIT {limit}"

        view = f"tmp_{uuid4().hex}"

        con = self.connect()
        con.execute(text(f"CREATE TEMP VIEW {view} AS {q}"))

        return SearchResult(con, view)


def query_batch(
    q: str, con: Connection, bsize: int = DEFAULT_BSIZE, limit: Optional[int] = None
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


def query_count(q: str, con: Connection, limit: Optional[int] = None) -> int:
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
