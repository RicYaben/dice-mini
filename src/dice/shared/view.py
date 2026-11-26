from collections.abc import Generator, Iterator, Sequence
from dataclasses import dataclass
from uuid import uuid4
from warnings import deprecated

import pandas as pd
from sqlalchemy import Connection, CursorResult, RowMapping, TextClause
from sqlmodel import func, select, text

from ._query import query


@dataclass
class Page:
    items: Sequence[RowMapping]
    number: int


@dataclass
class Paginator:
    con: Connection
    stmt: TextClause
    total: int
    current: int
    psize: int

    def fetch(self, p: int) -> Page:
        items = (
            self.con.execute(
                self.stmt,
                parameters={
                    "_limit": self.psize,
                    "_offset": p * self.psize,
                },
            )
            .mappings()
            .all()
        )

        return Page(items, p)

    def next(self) -> Page | None:
        offset = self.current * self.psize
        if offset >= self.total:
            return

        page = self.fetch(self.current)
        if len(page.items) > 0:
            return

        self.current += 1
        return page


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

    def stream(self, bsize: int | None = None) -> Generator[Sequence[RowMapping]]:
        m = self._c.mappings()
        if bsize:
            m = m.yield_per(bsize)
        yield from m.partitions(bsize)

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
        c = self.con.execute(text(q))
        return Cursor(c)

    def paginate(self, q: str, psize: int) -> Paginator:
        stmt = text(q)
        total = self.con.execute(select(func.count()).select_from(stmt)).scalar_one()

        return Paginator(
            con=self.con,
            stmt=stmt,
            total=total,
            current=0,
            psize=psize,
        )

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
