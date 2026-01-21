from abc import ABC
from typing import Any, Callable
import duckdb
import pandas as pd

import os
import copy

from enum import Enum

class OnConflict(Enum):
    FORCE = "force"        # blind append
    # ERROR = "error" <- not implemented
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

def table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    return bool(
        con.execute(
            """
            SELECT 1
            FROM information_schema.tables
            WHERE table_name = ?
            """,
            [table],
        ).fetchone()
    )

def create_table_from_df(
    con: duckdb.DuckDBPyConnection,
    table: str,
    df: pd.DataFrame,
) -> None:
    con.register("src_df", df)
    con.execute(f"""
        CREATE TABLE "{table}" AS
        SELECT * FROM src_df
    """)

def pk_eq(lhs: str, rhs: str, pkey: tuple[str, ...]) -> str:
    return " AND ".join(f"{lhs}.{c} = {rhs}.{c}" for c in pkey)

def pk_in(subquery: str, pkey: tuple[str, ...]) -> str:
    cols = ", ".join(pkey)
    return f"({cols}) IN ({subquery})"

def merge_sql(
    table: str,
    pkey: tuple[str, ...],
    cols: list[str],
    policy: OnConflict,
) -> str:
    non_pk = [c for c in cols if c not in pkey]

    if policy == OnConflict.FORCE:
        return f"""
        INSERT INTO "{table}" ({", ".join(cols)})
        SELECT {", ".join(cols)} FROM src_df
        """

    if policy == OnConflict.IGNORE:
        return f"""
        INSERT INTO "{table}" ({", ".join(cols)})
        SELECT {", ".join(f"s.{c}" for c in cols)}
        FROM src_df AS s
        WHERE NOT EXISTS (
            SELECT 1
            FROM "{table}" AS t
            WHERE {pk_eq("t", "s", pkey)}
        )
        """

    if policy == OnConflict.UPDATE:
        if not non_pk:
            return ""  # nothing to update

        return f"""
        UPDATE "{table}" AS t
        SET {", ".join(f"{c} = s.{c}" for c in non_pk)}
        FROM src_df AS s
        WHERE {pk_eq("t", "s", pkey)}
        """

    if policy == OnConflict.UPSERT:
        # DuckDB-style UPSERT = UPDATE + INSERT
        return f"""
        UPDATE "{table}" AS t
        SET {", ".join(f"{c} = s.{c}" for c in non_pk)}
        FROM src_df AS s
        WHERE {pk_eq("t", "s", pkey)};

        INSERT INTO "{table}" ({", ".join(cols)})
        SELECT {", ".join(f"s.{c}" for c in cols)}
        FROM src_df AS s
        WHERE NOT EXISTS (
            SELECT 1
            FROM "{table}" AS t
            WHERE {pk_eq("t", "s", pkey)}
        )
        """

    if policy == OnConflict.REPLACE:
        return f"""
        DELETE FROM "{table}"
        WHERE {pk_in(f"SELECT {', '.join(pkey)} FROM src_df", pkey)};

        INSERT INTO "{table}" ({", ".join(cols)})
        SELECT {", ".join(cols)} FROM src_df
        """

    raise ValueError(f"Unknown conflict policy: {policy}")

def inserter(
    con: duckdb.DuckDBPyConnection,
    table: str,
    *,
    pkey: tuple[str, ...] = ("id",),
    policy: OnConflict = OnConflict.UPSERT,

) -> Callable[[pd.DataFrame], None]:

    def cb(df: pd.DataFrame) -> None:
        if df.empty:
            return
        
        if not table_exists(con, table):
            create_table_from_df(con, table, df)
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

