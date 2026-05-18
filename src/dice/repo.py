import pandas as pd
import warnings
import logging

from typing import Any, Generator, Callable, Optional, Sequence
from sqlmodel import Session, text
from sqlalchemy import Connection, Row

from dice.health import HealthMonitor
from dice.config import DEFAULT_BSIZE
from dice.database import Connector, insert_or_ignore
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
        with self.connect() as c:
            res = c.execute(text(q))
            cols = [c[0] for c in res.cursor.description]  # type: ignore

            while True:
                if rows := res.fetchmany(bsize):
                    for r in rows:
                        yield dict(zip(cols, r))
                    continue
                break

    def stream(self, q: str) -> Generator[dict]:
        for batch in self.simple_query(q):
            for record in batch:
                yield record

    def query_batch(
        self, q: str, bsize: int = DEFAULT_BSIZE, norm = normalize_data, limit: Optional[int] = None
    ) -> Generator[pd.DataFrame, None, None]:
        with self.connect() as con:
            norm = norm if norm else lambda x: x
            for batch in query_batch(q, con, bsize, limit):
                df = pd.DataFrame.from_records(batch)
                yield norm(df)

    def query(
        self, q: str, bsize: int = DEFAULT_BSIZE, norm=normalize_data, limit: Optional[int] = None 
    ) -> tuple[int, Generator[pd.DataFrame, None, None]]:
        with self.connect() as con:
            d = query_count(q, con, limit)
        gen = self.query_batch(q, bsize, norm, limit)
        return (d, gen)
    
def query_batch(q: str, con: Connection, bsize: int = DEFAULT_BSIZE, limit: Optional[int] = None) -> Generator[Sequence, None, None]:
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
    d = (
        res[0] if (res := con.execute(text(dq)).fetchone()) else 0
    )
    return d

def new_repository(connector: Connector) -> Repository:
    return Repository(
        con=connector,
    )


