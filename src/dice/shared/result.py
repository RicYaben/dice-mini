from collections.abc import Generator, Iterator, Sequence
from uuid import uuid4

import pandas as pd
from sqlalchemy import Connection, CursorResult, RowMapping
from sqlmodel import text
from tqdm import tqdm

from dice.shared._query import query


# TODO: I would love to have a pagination option here
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
                yield from b
                bar.update(len(b))

    def all(self) -> list[dict]:
        res = self.query(f"SELECT * FROM {self.view}")
        cols = res.keys()

        return [dict(zip(cols, row)) for row in res.fetchall()]

    def one(self) -> dict:
        res = self.query(f"SELECT * FROM {self.view}")
        cols = res.keys()

        first = res.fetchone()
        if first is None:
            return {}

        return dict(zip(cols, first))

    def df(self, bsize: int | None = None) -> Iterator[pd.DataFrame] | pd.DataFrame:
        return pd.read_sql(
            text(f"SELECT * FROM {self.view}"), self.con, chunksize=bsize
        )

    def count(self) -> int:
        return self.query(f"SELECT COUNT(*) FROM {self.view}").scalar_one()

    def where(self, fields: list[str] | None = None, **clauses) -> "SearchResult":
        if fields is None:
            fields = ["*"]

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
