from collections import defaultdict
from itertools import chain
from typing import Any, Generator, Callable

import pandas as pd
import duckdb
import ujson
import copy
import uuid
import queue
import gc

from dice.models import HostTag, Source, Label, Fingerprint, FingerprintLabel, Tag
from dice.helpers import new_collection
from dice.config import DEFAULT_BSIZE, DATA_PREFIX

import warnings

warnings.simplefilter(action='ignore', category=UserWarning)

type RecordsWrapper = Callable[[Any], pd.DataFrame]

def with_items(*objs) -> pd.DataFrame:
    return new_collection(*objs).to_df()

def save(con, name: str, df: pd.DataFrame, bsize: int = DEFAULT_BSIZE):
    if df.empty:
        return
    
    df.to_sql(name, con, if_exists="append", index=False, method="multi", chunksize=bsize)

# def save(con, name: str, df: pd.DataFrame, bsize: int = DEFAULT_BSIZE):
#     cols = list(df.columns)
#     placeholders = ", ".join(["?"] * len(cols))
#     sql = f"INSERT INTO {name} ({', '.join(cols)}) VALUES ({placeholders})"

#     print(f"saving {name}...")
#     cursor = con.cursor()
#     for start in range(0, len(df), bsize):
#         batch = df.iloc[start:start+bsize]
#         cursor.executemany(sql, batch.itertuples(index=False, name=None))
#         con.commit()
#     print(f"saved")

def normalize_data(df: pd.DataFrame, prefix: str="") -> pd.DataFrame:
    # cannot parse
    if not df.iloc[0].get("data", None):
        return df

    parsed = df["data"].map(ujson.loads)
    rdf = pd.json_normalize(parsed.tolist(), max_level=0).add_prefix(prefix)
    norm = pd.concat([
        df.drop(columns=["data"]).reset_index(drop=True),
        rdf.reset_index(drop=True)
    ], axis=1)
    return norm

def normalize_zgrab2_records(df: pd.DataFrame, prefix: str="") -> pd.DataFrame:
    parsed = df["data"].apply(ujson.loads)

    # Flatten the 'result' dict
    rdf = pd.json_normalize(parsed.tolist(), max_level=1)
    rdf.columns = rdf.columns.str.removeprefix("result.")
    rdf = rdf.add_prefix(prefix)

    # Concatenate original df (without 'data') and flattened result columns
    norm = pd.concat([
        df.drop(columns=["data"]).reset_index(drop=True),
        rdf.reset_index(drop=True)
    ], axis=1)

    return norm

def normalize_records(df: pd.DataFrame) -> pd.DataFrame:
    """each record contains a source_name and an id, that is enough"""
    match df.iloc[0].get("source_name"):
        case "zgrab2":
            return normalize_zgrab2_records(df, DATA_PREFIX)
        case _:
            return normalize_data(df, DATA_PREFIX)

def normalize_fingerprints(df: pd.DataFrame) -> pd.DataFrame: 
    return normalize_data(df, DATA_PREFIX)

class Connector:
    def __init__(self, db_path: str | None, name: str = "-", read_only: bool = False, config: dict = {}) -> None:
        self.name = name
        self.db_path: str = db_path if db_path else ":memory:"
        self.readonly = read_only
        self.config = config

        self.con: duckdb.DuckDBPyConnection | None = None
        self._extensions()
        self._pragmas()

        if not read_only:
            self._macros()

    def get_connection(self) -> duckdb.DuckDBPyConnection:
        if not self.con:
            self.con = self.new_connection()
        return self.con
    
    def new_connection(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(self.db_path, read_only=self.readonly) #config=self.config)
    
    def with_connection(self, name: str) -> 'Connector':
        return Connector(self.db_path, name, self.readonly, self.config)
    
    def _extensions(self) -> None:
        conn = self.get_connection()
        conn.execute("INSTALL inet")
        conn.execute("LOAD inet")
    
    def _pragmas(self) -> None:
        conn = self.get_connection()
        conn.execute("PRAGMA temp_directory='./duckdb_tmp';")
        conn.execute("PRAGMA threads=8;")
        conn.execute("PRAGMA memory_limit='10GB';")
        conn.execute("PRAGMA max_memory='10GB';")
        conn.execute("PRAGMA preserve_insertion_order=FALSE;")
        conn.execute("PRAGMA checkpoint_threshold='100GB';") # very importante

    def _macros(self) -> None:
        conn = self.get_connection()
        conn.execute("""
            CREATE OR REPLACE MACRO network_from_cidr(cidr_range) AS (
                cast(string_split(string_split(cidr_range, '/')[1], '.')[1] as bigint) * (256 * 256 * 256) +
                cast(string_split(string_split(cidr_range, '/')[1], '.')[2] as bigint) * (256 * 256      ) +
                cast(string_split(string_split(cidr_range, '/')[1], '.')[3] as bigint) * (256            ) +
                cast(string_split(string_split(cidr_range, '/')[1], '.')[4] as bigint)
            );
        """)

        conn.execute("""
            CREATE OR REPLACE MACRO broadcast_from_cidr(cidr_range) AS (
                cast(string_split(string_split(cidr_range, '/')[1], '.')[1] as bigint) * (256 * 256 * 256) +
                cast(string_split(string_split(cidr_range, '/')[1], '.')[2] as bigint) * (256 * 256      ) +
                cast(string_split(string_split(cidr_range, '/')[1], '.')[3] as bigint) * (256            ) +
                cast(string_split(string_split(cidr_range, '/')[1], '.')[4] as bigint)) + 
                cast(pow(256, (32 - cast(string_split(cidr_range, '/')[2] as bigint)) / 8) - 1 as bigint
            );
        """)

        conn.execute("""
            CREATE OR REPLACE MACRO ip_within_cidr(ip, cidr_range) AS (
                network_from_cidr(ip || '/32') >= network_from_cidr(cidr_range) AND network_from_cidr(ip || '/32') <= broadcast_from_cidr(cidr_range)
            );       
        """)

class Repository:
    _collection: duckdb.DuckDBPyRelation

    def __init__(self, connector: Connector, q_size=2, writers: int=2, processors: int=2) -> None:
        self.connector = connector
        self.writers = writers
        self.processors = processors
        self._queue = queue.Queue(maxsize=q_size)

    def _query(self, table: str, **kwargs) -> pd.DataFrame:
        q = f"SELECT * FROM {table}"
        clauses = []
        params = []

        for k, val in kwargs.items():
            if isinstance(val, list):
                placeholders = ", ".join(["?"] * len(val))
                clauses.append(f"{k} IN ({placeholders})")
                params.extend(val)
            else:
                clauses.append(f"{k} = ?")
                params.append(val)

        if clauses:
            q += " WHERE " + " AND ".join(clauses)

        return self.get_connection().execute(q, params).df()
    
    def _rebuild_records_view(self) -> None:
        tables = self.get_connection().execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_name LIKE 'records_%'
                AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """).fetchall()

        grouped = defaultdict(list)
        for (tname,) in tables:
            prefix = "_".join(tname.split("_")[:-1])
            grouped[prefix].append(tname)

        for (prefix, tnames) in grouped.items():
            union_sql = " UNION ALL ".join([f'SELECT * FROM "{t}"' for t in tnames])
            try:
                self.get_connection().execute(f'CREATE OR REPLACE VIEW "{prefix}" AS {union_sql}')
            except Exception as e:
                print(f"failed to rebuild records view: {prefix}")
                raise e

    def get_connection(self) -> duckdb.DuckDBPyConnection:
        return self.connector.get_connection()

    def get_fingerprints(self, *host: str, normalize: bool= False, **kwargs) -> pd.DataFrame:
        'query the repo for records'
        # select all hosts
        if len(host):
            kwargs["host"] = list(host)
        df = self._query("fingerprints", **kwargs)
        if normalize:
            return normalize_fingerprints(df)
        return df
    
    def get_records(self, *host: str, source: str="zgrab2", normalize: bool= False, prefix: str | None ="records",**kwargs) -> pd.DataFrame:
        'query the repo for (ALL) records'
        # select all hosts
        if len(host):
            kwargs["ip"] = list(host)
        
        df = self._query("_".join(filter(None, [prefix,source])), **kwargs)
        if normalize:
            return normalize_records(df)
        return df
    
    def get_tags(self, *host: str, **kwargs) -> pd.DataFrame:
        if len(host):
            kwargs["host"] = list(host)
        df = self._query("host_tags", **kwargs)
        return df
    
    def summary(self) -> dict:
        'returns a brief summary of the database contents'
        con = self.get_connection()
        fpd = con.execute("""
            SELECT COUNT(DISTINCT host)
            FROM fingerprints       
        """).fetchone()

        lbld = con.execute("""
            SELECT COUNT(DISTINCT fp.host)
            FROM fingerprints fp
            JOIN fingerprint_labels fl ON fl.fingerprint_id = fp.id
        """).fetchone()

        summary = {}

        if fpd is not None:
            summary["fingerprinted"] = fpd[0]

        if lbld is not None:
            summary["labelled"] = lbld[0]
        
        return summary
    
    # ---
    def _peek(self, src_gen: Generator[pd.DataFrame, None, None]) -> pd.DataFrame | None:
        try:
            return next(src_gen)
        except StopIteration:
            return
        
    def _fmt(self, df: pd.DataFrame, oc: list[str] = [], ic: list[str] = []) -> pd.DataFrame:
        for col in oc:
            df[col] = df[col].map(
                lambda x: ujson.dumps(x) if isinstance(x, (dict, list)) else x
            )

        for col in ic:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64", copy=False)
            
        df["id"] = [uuid.uuid4().hex for _ in range(len(df))]

        return df
    
    def _get_fmt_columns(self, df: pd.DataFrame) -> tuple[list[str], list[str]]:
        oc = []

        n = min(50, len(df))
        print(f"sampling source with {n}/{len(df)}")
        s = df.sample(n)

        for col in df.select_dtypes(include=["object"]).columns:
            if s[col].apply(lambda x: isinstance(x, (dict, list))).any():
                oc.append(col)

        ic = list(df.select_dtypes(include=["int", "Int64", "Int32"]).columns)
        return oc, ic
         
    def add_sources(self, *sources: Source):
        for src in sources:
            self.add_source(src)

    def add_source(self, src: Source):
        print(f"adding source: {src.name}")

        c_gen = src.load()
        peek = self._peek(c_gen)
        if peek is None:
            print(f"empty source: {src.name}")
            return

        oc, ic = self._get_fmt_columns(peek)
        tab = f"records_{src.name}_{src.id}"
        con = self.get_connection()

        print(f"inserting {src.name} records into {tab}")
        
        for i, c in enumerate(chain([peek], c_gen), start=1):
            save(con, tab, self._fmt(c, oc, ic), src.batch_size)
            # chain and other iterators keep a reference to the items, preventing gc
            if i % 10 == 0:
                print(f"[{src.name}] {i} batches inserted...")
                gc.collect()

            del c
        self._rebuild_records_view()
    # ---

    def _add_items(self, *items: Any, name: str) -> None:
        save(self.get_connection(), df=with_items(*items), name=name)

    def add_hosts(self, hosts: pd.DataFrame) -> None:
        save(self.get_connection(), df=hosts, name="hosts")
    
    def add_labels(self, *label: Label) -> None:
        'create new labels in the database'
        self._add_items(*label, name="labels")

    def add_tags(self, *tag: Tag) -> None:
        self._add_items(*tag, name="tags")

    def fingerprint(self, *fingerprint: Fingerprint) -> None:
        'fingerprint hosts. The dataframe is a collection of fingerprints'
        self._add_items(*fingerprint, name="fingerprints")

    def label(self, *fp_lab: FingerprintLabel) -> None:
        'label fingerprint. The dataframe is a dict of label_id and host_id'
        self._add_items(*fp_lab, name="fingerprint_labels")

    def tag(self, *host_tag: HostTag) -> None:
        'tag a host'
        self._add_items(*host_tag, name="host_tags")

    def with_view(self, name: str, q: str) -> 'Repository':
        c = copy.copy(self)
        con = c.get_connection()
        con.sql(f"CREATE OR REPLACE VIEW {name} AS {q}")
        c._collection = con.sql(f"SELECT * FROM {name}")
        return c

    def collect(self) -> pd.DataFrame:
        return self._collection.df()
    
    def query(self, q: str, bsize=DEFAULT_BSIZE) -> Generator[Any, None, None]:
        cursor = self.get_connection().execute(q)
        cols = [c[0] for c in cursor.description] # type: ignore

        while True:
            if rows := cursor.fetchmany(bsize):
                for r in rows:
                    yield dict(zip(cols, r))
                continue
            break

    def query_batch(self, q, normalize: bool=True, bsize=DEFAULT_BSIZE) -> Generator[pd.DataFrame, None, None]:
        cursor = self.get_connection().execute(q)
        norm = normalize_records if normalize else lambda x: x
        
        for b in cursor.fetch_record_batch(bsize):
            yield norm(b.to_pandas())

    def queryb(self, q, normalize: bool=True, bsize=DEFAULT_BSIZE) -> tuple[int, Generator[Any, None, None]]:
        dq = f"WITH ct AS ({q}) SELECT COUNT(*) AS rows FROM ct;"
        d = res[0] if (res:=self.get_connection().execute(dq).fetchone()) else 0
        return d, self.query_batch(q, normalize, bsize)

def load_repository(sources: list[Source] = [], db: str|None=None, read_only: bool = False) -> Repository:
    '''
    Create a repository from a list of sources.
    - db: name of the database - if not set, load in memory
    '''
    connector = Connector(db, read_only=read_only)
    repo = Repository(connector)
    repo.add_sources(*sources)
    return repo
