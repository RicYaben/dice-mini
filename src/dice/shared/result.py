import pandas as pd

from uuid import uuid4
from typing import Generator, Optional, Sequence

from sqlalchemy import Connection, CursorResult, RowMapping
from sqlmodel import text
from tqdm import tqdm

from dice.shared._query import query


class SearchResult:
    def __init__(
        self,
        con: Connection,
        view: str,
    ):
        self.con = con
        self.view = view

    def query(self, q: str) -> CursorResult:
        return self.con.execute(text(q))
    
    def batch(self, bsize: int = 50_000) -> Generator[Sequence[RowMapping]]:
        res = self.query(f"SELECT * FROM {self.view}").mappings()
        while rows := res.fetchmany(bsize):
            yield rows
            
    def stream(self) -> Generator[RowMapping]:
        with tqdm(total=self.count()) as bar:
            for b in self.batch():
                for r in b:
                    yield r
                bar.update(len(b))

    def all(self) -> list[dict]:
        res = self.query(f"SELECT * FROM {self.view}")
        cols = res.keys()

        return [
            dict(zip(cols, row))
            for row in res.fetchall()
        ]

    def df(self, bsize: Optional[int] = None) -> Generator[pd.DataFrame] | pd.DataFrame:
        return pd.read_sql(
            text(f"SELECT * FROM {self.view}"),
            self.con,
            chunksize=bsize
        )

    def count(self) -> int:
        return self.query(
            f"SELECT COUNT(*) FROM {self.view}"
        ).scalar_one()

    def where(self, fields: list[str] =["*"], **clauses) -> "SearchResult":
        new_view = f"tmp_{uuid4().hex}"

        q = f"CREATE TEMP VIEW {new_view} AS {query(self.view, fields, **clauses)}"
        self.con.execute(text(q))

        return SearchResult(self.con, new_view)

    def close(self):
        self.con.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __iter__(self) -> Generator[RowMapping]:
        yield from self.stream()

    def __len__(self) -> int:
        return self.count()
    
    def __bool__(self) -> bool:
        return self.count() > 0