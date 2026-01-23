from itertools import chain
from typing import Generator, Optional

from dice.loaders import walk
from dice.models import Source, Cursor

import uuid
import duckdb
import ujson
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def counter_get_or_create(
    con: duckdb.DuckDBPyConnection,
    name: str,
    start: int = 0
) -> int:
    con.execute("""
        INSERT INTO counters (name, value)
        VALUES (?, ?)
        ON CONFLICT (name) DO NOTHING
    """, (name, start))

    return con.execute("""
        SELECT value FROM counters WHERE name = ?
    """, (name,)).fetchone()[0] # type: ignore

def counter_update(
    con: duckdb.DuckDBPyConnection,
    name: str,
    step: int = 1
) -> int:
    result = con.execute("""
        UPDATE counters
        SET value = value + ?
        WHERE name = ?
        RETURNING value
    """, (step, name)).fetchone()

    if result is None:
        raise KeyError(f"Counter '{name}' does not exist")

    return result[0]

class Watcher:
    _gen: Optional[Generator[pd.DataFrame, None, None]]
    _peek: Optional[pd.DataFrame]
    _peeked: bool = False

    _oc: list[str]
    _ic: list[str]

    def __init__(self, table: str, src: Source, cursor: Cursor) -> None:
        self.table = table
        self.src = src
        self.cursor = cursor
    
    @property
    def peek(self) -> pd.DataFrame | None:
        if not self._gen:
            self.load()

        if self._peeked:
            return self._peek
        
        try:
            assert isinstance(self._gen, Generator)
            p = next(self._gen)
            self._peek = p
        except StopIteration:
            pass

        self._peeked = True
        return self._peek
    
    @property
    def columns(self)  -> tuple[list[str], list[str]]:
        if self._oc or self._ic:
            return (self._oc, self._ic)
        
        if self.empty():
            raise ValueError("unable to get columns: empty source") 

        p = self.peek
        assert isinstance(p, pd.DataFrame)

        n = min(1000, len(p))
        logger.debug(f"polling source with {n}/{len(p)}")
        s = p.sample(n)

        # object cols
        oc = []
        for col in p.select_dtypes(include=["object"]).columns:
            if s[col].apply(lambda x: isinstance(x, (dict, list))).any():
                oc.append(col)

        # numeric cols
        ic = list(p.select_dtypes(include=["number"]).columns)

        self._oc = oc
        self._ic = ic
        return (oc, ic)
    
    def exists(self) -> bool:
        if not next(walk(self.src.path), None):
            return False
        return True
    
    def load(self) -> Generator[pd.DataFrame, None, None]:
        if self._gen:
            return self._gen
        
        # TODO: this is not very optimal, it would be better if we could
        # send the cursor to the loader, but that kills the globs, because we don't know
        # which file the cursor is tracking
        data = self.src.load()
        for _ in range(self.cursor.index):
            next(data, None)

        self._gen = data
        return data

    def reset(self):
        self._gen = None
        self._peek = None
        self._peeked = False
        self.cursor.reset()

    def format_columns(self, df: pd.DataFrame, oc, ic: list[str]) -> pd.DataFrame:
        # convert to string dict and list cols
        for col in oc:
            df[col] = df[col].map(
                lambda x: ujson.dumps(x) if isinstance(x, (dict, list)) else x
            )

        # convert to int64 numeric cols
        for col in ic:
            df[col] = pd.to_numeric(df[col], errors="coerce", dtype_backend="pyarrow", downcast="float")

        df["id"] = [uuid.uuid4().hex for _ in range(len(df))]
        return df

    def consume(self) -> Generator[pd.DataFrame, None, None]:
        data = self.load()

        p = self.peek
        assert isinstance(p, pd.DataFrame)

        oc, ic = self.columns
        for c in chain(p, data):
            assert isinstance(c, pd.DataFrame)

            fmt = self.format_columns(c, oc, ic)
            yield fmt

            self.cursor.update()

        # reset the peek and generator
        self.reset()

    def empty(self) -> bool:
        p = self.peek
        return p is not None and not p.empty
    
    def check(self):
        if not self.exists():
            raise ValueError(f"source not found: {self.src.path}")
        if self.empty():
            raise ValueError(f"empty source: {self.src.name}")


def new_watcher(table: str, src: Source, cursor: Cursor) -> Watcher:
    return Watcher(table, src, cursor)