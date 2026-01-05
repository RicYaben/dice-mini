from abc import ABC
from typing import Any, Callable
import duckdb
import pandas as pd

import os
import copy

from dice.config import DEFAULT_BSIZE

class VoidStore(ABC):
    def store(self, _): ...
    def close(self): ...

class LocalStore:
    def __init__(self, dir: str) -> None:
        if os.path.exists(dir) and not os.path.isdir(dir):
            raise Exception(f"path exists, but is not a directory: {dir}")
        self.dir = dir
        self.buf = None

    def open(self, fname: str) -> 'LocalStore':
        buf = open(os.path.join(self.dir, fname), mode="+a")
        c = copy.copy(self)
        c.buf = buf
        return c

    def store(self, df: pd.DataFrame) -> None:
        df.to_json(self.buf, orient="records", lines=True, index=False)
        
    def close(self):
        if self.buf is not None:
            self.buf.close()

def new_local_store(dir: str):
    return LocalStore(dir)

def make_store(dir: str | None = None, fname: str | None = None):
    if not (dir and fname):
        return VoidStore()
    s = LocalStore(dir).open(fname)
    return s

def new_inserter(
    con: duckdb.DuckDBPyConnection,
    table: str,
    force: bool = False,
    bsize: int = DEFAULT_BSIZE,
) -> Callable[[pd.DataFrame], None]:
    'returns a callback to insert chunked dfs into a database'

    def insert(df):
        df.to_sql(
            table, con, if_exists="append", index=False, method="multi", chunksize=bsize
        )

    def cb(df: pd.DataFrame) -> None:
        if df.empty:
            return

        # ---- FORCE MODE: insert everything ----
        if force:
            insert(df)
            return
        
        pkey = "id"

        # ---- Ensure PK hash column exists ----
        if pkey not in df.columns:
            raise ValueError(f"DataFrame must contain a '{pkey}' primary-key hash column")

        try:
            values_clause = ", ".join(f"('{h}')" for h in df[pkey])

            query = f"""
            WITH batch_hashes({pkey}) AS (
                VALUES {values_clause}
            )
            SELECT {pkey}
            FROM batch_hashes
            WHERE {pkey} NOT IN (
                SELECT {pkey} FROM {table}
            )
            """

            # fetch hashes that are not in the DB
            new_hashes_df = con.execute(query).fetchdf()

        except duckdb.CatalogException:
            # table does not exist → insert everything
            insert(df)
            return

        # ---- Filter only brand-new rows ----
        new_hashes = set(new_hashes_df[pkey])
        new_rows = df[df[pkey].isin(new_hashes)]

        if not new_rows.empty:
            insert(new_rows)
    return cb

class Store:

    def __init__(self, store_conf: dict[str, str], insert_conf: dict[str, Any]) -> None:
        self.sconf = store_conf
        self.iconf = insert_conf

    def __enter__(self) -> 'Store':
        self.storage = make_store(**self.sconf)
        self.inserter = new_inserter(**self.iconf)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def store(self, df: pd.DataFrame) -> None:
        self.storage.store(df)

    def insert(self, df: pd.DataFrame) -> None:
        self.inserter(df)

    def save(self, df: pd.DataFrame) -> None:
        self.store(df)
        self.insert(df)

    def close(self) -> None: 
        self.storage.close()

def store(store_conf: dict[str, str], insert_conf: dict[str, str]) -> Store:
    s = Store(store_conf=store_conf, insert_conf=insert_conf)
    return s

