from typing import Any, Generator, Callable
from sqlmodel import Session, text
from sqlalchemy import Connection

from dice.health import HealthMonitor
from dice.config import DEFAULT_BSIZE
from dice.database import Connector, insert_or_ignore

import pandas as pd
import warnings
import logging

from dice.helpers import normalize_data

warnings.simplefilter(action="ignore", category=UserWarning)

logger = logging.getLogger(__name__)

type RecordsWrapper = Callable[[Any], pd.DataFrame]

class Repository:
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

    def insert(self, items: list[Any], policy=insert_or_ignore, con: Connection | None = None):
        if not items:
            return

        model = type(items[0])
        if not con:
            con = self.connect()

        with Session(con) as s:
            policy(s, model, items)
            s.flush()

    def simple_query(
        self, q: str, bsize: int = DEFAULT_BSIZE
    ) -> Generator[dict, None, None]:
        with self.connect() as con:
            res = con.execute(text(q))
            cols = [c[0] for c in res.cursor.description]  # type: ignore

            while True:
                if rows := res.fetchmany(bsize):
                    for r in rows:
                        yield dict(zip(cols, r))
                    continue
                break

    def query_batch(
        self, q: str, bsize: int = DEFAULT_BSIZE, norm = normalize_data
    ) -> Generator[pd.DataFrame, None, None]:
        with self.connect() as con:
            res = con.execute(text(q)).mappings()
            norm = norm if norm else lambda x: x

            while rows := res.fetchmany(bsize):
                yield norm(pd.DataFrame.from_records(rows)) # type: ignore

    def query_count(self, q: str) -> int:
        dq = f"WITH ct AS ({q}) SELECT COUNT(*) AS rows FROM ct;"
        with self.connect() as con:
            d = (
                res[0] if (res := con.execute(text(dq)).fetchone()) else 0
            )
            return d

    def query(
        self, q: str, bsize: int = DEFAULT_BSIZE, norm=normalize_data
    ) -> tuple[int, Generator[pd.DataFrame, None, None]]:
        d = self.query_count(q)
        gen = self.query_batch(q, bsize, norm)
        return (d, gen)

def new_repository(connector: Connector) -> Repository:
    return Repository(
        con=connector,
    )


