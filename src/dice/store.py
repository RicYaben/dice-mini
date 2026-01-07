from abc import ABC
from typing import Any, Callable
import duckdb
import pandas as pd

import os
import copy

from enum import Enum

class OnConflict(Enum):
    FORCE = "force"        # blind append
    ERROR = "error"        # raise if conflict
    IGNORE = "ignore"      # insert only new rows
    UPDATE = "update"      # update existing rows
    UPSERT = "upsert"      # update + insert
    REPLACE = "replace"    # delete + insert (full overwrite)

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

def merge_sql(
    table: str,
    pkey: str,
    cols: list[str],
    policy: OnConflict,
) -> str:
    non_pk = [c for c in cols if c != pkey]

    if policy == OnConflict.FORCE:
        # dumb
        return f"""
        INSERT INTO {table} ({", ".join(cols)})
        SELECT {", ".join(cols)} FROM src_df
        """

    if policy == OnConflict.ERROR:
        # pure insert, let DB error
        return f"""
        INSERT INTO {table} ({", ".join(cols)})
        SELECT {", ".join(cols)} FROM src_df
        """

    if policy == OnConflict.IGNORE:
        # insert only new
        return f"""
        MERGE INTO {table} AS t
        USING src_df AS s
        ON t.{pkey} = s.{pkey}
        WHEN NOT MATCHED THEN
          INSERT ({", ".join(cols)})
          VALUES ({", ".join(f"s.{c}" for c in cols)})
        """

    if policy == OnConflict.UPDATE:
        # update conflicting
        return f"""
        MERGE INTO {table} AS t
        USING src_df AS s
        ON t.{pkey} = s.{pkey}
        WHEN MATCHED THEN
          UPDATE SET {", ".join(f"{c} = s.{c}" for c in non_pk)}
        """

    if policy == OnConflict.UPSERT:
        # update and merge
        return f"""
        MERGE INTO {table} AS t
        USING src_df AS s
        ON t.{pkey} = s.{pkey}
        WHEN MATCHED THEN
          UPDATE SET {", ".join(f"{c} = s.{c}" for c in non_pk)}
        WHEN NOT MATCHED THEN
          INSERT ({", ".join(cols)})
          VALUES ({", ".join(f"s.{c}" for c in cols)})
        """

    if policy == OnConflict.REPLACE:
        # replace all
        return f"""
        DELETE FROM {table}
        WHERE {pkey} IN (SELECT {pkey} FROM src_df);

        INSERT INTO {table} ({", ".join(cols)})
        SELECT {", ".join(cols)} FROM src_df
        """

    raise ValueError(f"Unknown conflict policy: {policy}")


def inserter(
    con: duckdb.DuckDBPyConnection,
    table: str,
    *,
    pkey: str = "id",
    policy: OnConflict = OnConflict.UPSERT,
) -> Callable[[pd.DataFrame], None]:

    def cb(df: pd.DataFrame) -> None:
        if df.empty:
            return

        con.register("src_df", df)

        sql = merge_sql(
            table=table,
            pkey=pkey,
            cols=df.columns.tolist(),
            policy=policy,
        )

        con.execute(sql)

    return cb

class Store:

    def __init__(self, store_conf: dict[str, str], insert_conf: dict[str, Any]) -> None:
        self.sconf = store_conf
        self.iconf = insert_conf

    def __enter__(self) -> 'Store':
        self.storage = make_store(**self.sconf)
        self.inserter = inserter(**self.iconf)
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

