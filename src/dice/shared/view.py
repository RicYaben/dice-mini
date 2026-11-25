from collections.abc import Iterator, Sequence
from uuid import uuid4
from warnings import deprecated

import pandas as pd
from sqlalchemy import Connection, CursorResult, RowMapping
from sqlmodel import text

from ._query import query


class Cursor:
    def __init__(
        self,
        c: CursorResult,
    ) -> None:
        self._c = c

    def count(self) -> int:
        return self._c.rowcount

    def iter(self, bsize: int | None = None) -> Iterator[Sequence[RowMapping]]:
        res = self._c.mappings()
        return res.partitions(bsize)

    def all(self) -> Sequence[RowMapping]:
        return self._c.mappings().all()


class View:
    def __init__(
        self,
        con: Connection,
        table: str,
        bsize: int | None = None,
    ):
        self.con = con
        self.table = table
        self.bsize = bsize

    def exec(self, q: str) -> Cursor:
        return Cursor(self.con.execute(text(q)))

    def count(self) -> int:
        return self.exec(f"SELECT COUNT(*) FROM {self.table}").count()

    @deprecated("Pandas and dataframes will not be supported from 1.0 onwards")
    def df(self, bsize: int | None = None) -> Iterator[pd.DataFrame] | pd.DataFrame:
        return pd.read_sql(
            text(f"SELECT * FROM {self.table}"), self.con, chunksize=bsize
        )

    def where(self, fields: list[str] | None = None, **clauses) -> "View":
        if fields is None:
            fields = ["*"]

        new_view = f"tmp_{uuid4().hex}"

        q = f"CREATE TEMP VIEW {new_view} AS {query(self.table, fields, **clauses)}"
        self.con.execute(text(q))

        return View(self.con, new_view)

    def close(self):
        self.con.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
