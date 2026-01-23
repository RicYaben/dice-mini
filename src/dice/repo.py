from collections import defaultdict
from itertools import chain
from typing import Any, Generator, Callable, Optional
from tqdm import tqdm

import pandas as pd
import duckdb
import ujson
import copy
import uuid

from dice.models import Model, Source, M_REQUIRED
from dice.helpers import new_collection, new_host
from dice.config import DEFAULT_BSIZE, DATA_PREFIX
from dice.store import OnConflict, inserter, store, table_exists
from dice.events import Event, new_event

import warnings

warnings.simplefilter(action="ignore", category=UserWarning)

type RecordsWrapper = Callable[[Any], pd.DataFrame]
type HealthCheck = Callable[[Repository, Event], None]

E_SOURCE = "source"
E_LOAD = "load"
E_SANITY = "sanity"

def with_items(*objs) -> pd.DataFrame:
    return new_collection(*objs).to_df()

def normalize_data(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    # cannot parse
    if not df.iloc[0].get("data", None):
        return df

    parsed = df["data"].map(ujson.loads)
    rdf = pd.json_normalize(parsed.tolist(), max_level=0).add_prefix(prefix)
    norm = pd.concat(
        [df.drop(columns=["data"]).reset_index(drop=True), rdf.reset_index(drop=True)],
        axis=1,
    )
    return norm


def normalize_zgrab2_records(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    parsed = df["data"].apply(ujson.loads)

    # Flatten the 'result' dict
    rdf = pd.json_normalize(parsed.tolist(), max_level=1)
    rdf.columns = rdf.columns.str.removeprefix("result.")
    rdf = rdf.add_prefix(prefix)

    # Concatenate original df (without 'data') and flattened result columns
    norm = pd.concat(
        [df.drop(columns=["data"]).reset_index(drop=True), rdf.reset_index(drop=True)],
        axis=1,
    )

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
    def __init__(
        self,
        db_path: str | None,
        name: str = "-",
        read_only: bool = False,
        config: dict = {},
    ) -> None:
        self.name = name
        self.db_path: str = db_path if db_path else ":memory:"
        self.readonly = read_only
        self.config = config

        self.con: duckdb.DuckDBPyConnection | None = None

    def init_con(self, con, readonly: bool):
        self._extensions(con)
        self._pragmas(con)
        if not readonly:
            self._macros(con)

    def get_connection(self) -> duckdb.DuckDBPyConnection:
        if not self.con:
            self.con = self.new_connection()
        return self.con

    def new_connection(self) -> duckdb.DuckDBPyConnection:
        con = duckdb.connect(
            self.db_path, read_only=self.readonly
        )  # config=self.config)

        self.init_con(con, self.readonly)
        return con

    def with_connection(self, name: str) -> "Connector":
        return Connector(self.db_path, name, self.readonly, self.config)
    
    def attach(self, path: str, name: str) -> None:
        con = self.get_connection()
        con.execute(f"ATTACH '{path}' AS {name};")

    def copy(self, table: str, src: str) -> None:
        con = self.get_connection()
        con.execute(f"DROP TABLE IF EXISTS 'main.{table}'")
        con.execute(f"CREATE TABLE 'main.{table}' AS SELECT * FROM '{src}.{table}'")

    def _extensions(self, conn: duckdb.DuckDBPyConnection) -> None:
        conn.execute("INSTALL inet")
        conn.execute("LOAD inet")

    def _pragmas(self, conn: duckdb.DuckDBPyConnection) -> None:
        conn.execute("PRAGMA temp_directory='./duckdb_tmp';")
        conn.execute("PRAGMA threads=8;")
        conn.execute("PRAGMA memory_limit='10GB';")
        conn.execute("PRAGMA max_memory='10GB';")
        conn.execute("PRAGMA preserve_insertion_order=FALSE;")
        conn.execute("PRAGMA checkpoint_threshold='100GB';")  # very importante

    def _macros(self, conn: duckdb.DuckDBPyConnection) -> None:
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

        conn.execute("""
            CREATE OR REPLACE FUNCTION inet_aton(ip) AS (
                cast(
                split_part(ip, '.', 1)::UINTEGER * 16777216 +
                split_part(ip, '.', 2)::UTINYINT * 65536 +
                split_part(ip, '.', 3)::UTINYINT * 256 +
                split_part(ip, '.', 4)::UTINYINT as UINTEGER)
            );
        """)

def new_connector(db: str, readonly: bool = False, name: str = "-") -> Connector:
    return Connector(db, read_only=readonly, name=name)


class Repository:
    _collection: duckdb.DuckDBPyRelation

    def __init__(
        self, 
        connector: Connector,
        monitor: 'Monitor',
    ) -> None:
        self.connector = connector
        self.monitor = monitor
        self.monitor.initialize(self)

    def _query(self, table: str, **kwargs) -> pd.DataFrame:
        q = f"SELECT * FROM '{table}'"
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
            
    def get_connection(self) -> duckdb.DuckDBPyConnection:
        return self.connector.get_connection()
    
    def set_sources_path(self, path: str | None) -> 'Repository':
        self._sources_path = path
        return self

    def get_fingerprints(
        self, *host: str, normalize: bool = False, **kwargs
    ) -> pd.DataFrame:
        "query the repo for records"
        # select all hosts
        if len(host):
            kwargs["host"] = list(host)
        df = self._query("fingerprints", **kwargs)
        if normalize:
            return normalize_fingerprints(df)
        return df

    def get_records(
        self,
        *host: str,
        source: str = "zgrab2",
        normalize: bool = False,
        prefix: str | None = "records",
        **kwargs,
    ) -> pd.DataFrame:
        "query the repo for (ALL) records"
        # select all hosts
        if len(host):
            kwargs["ip"] = list(host)

        df = self._query("_".join(filter(None, [prefix, source])), **kwargs)
        if normalize:
            return normalize_records(df)
        return df

    def get_tags(self, *host: str, **kwargs) -> pd.DataFrame:
        if len(host):
            kwargs["host"] = list(host)
        df = self._query("host_tags", **kwargs)
        return df

    def summary(self) -> dict:
        "returns a brief summary of the database contents"
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
    def _peek(
        self, src_gen: Generator[pd.DataFrame, None, None]
    ) -> pd.DataFrame | None:
        try:
            return next(src_gen)
        except StopIteration:
            return

    def _fmt(
        self, df: pd.DataFrame, oc: list[str] = [], ic: list[str] = []
    ) -> pd.DataFrame:
        
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

    def _get_fmt_columns(self, df: pd.DataFrame) -> tuple[list[str], list[str]]:
        n = min(1000, len(df))
        print(f"polling source with {n}/{len(df)}")
        s = df.sample(n)

        # object cols
        oc = []
        for col in df.select_dtypes(include=["object"]).columns:
            if s[col].apply(lambda x: isinstance(x, (dict, list))).any():
                oc.append(col)

        # numeric cols
        ic = list(df.select_dtypes(include=["number"]).columns)
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

        con = self.get_connection()

        oc, ic = self._get_fmt_columns(peek)
        tab = f"records_{src.name}_{src.id}"

        insert_conf={
            "con": con, 
            "table": tab, 
            "policy":OnConflict.FORCE
        }

        store_conf={
            "dir": self._sources_path, 
            "fname": src.name
        }

        with store(store_conf, insert_conf) as s:
            print(f"inserting {src.name} records into {tab} ({src._batch_size}/b)")
            for c in tqdm(chain([peek], c_gen)):
                d = self._fmt(c, oc, ic)
                s.save(d)
                del c

        summary = {
            "table": tab,
            "iconf": insert_conf,
            "sconf": store_conf,
        }
        event = new_event(E_SOURCE, summary)
        self.monitor.synchronize(event)

    # create or update
    def create(
        self, 
        *items: Any,
        policy: OnConflict = OnConflict.IGNORE, # update, ignore, error
    ) -> None:
        if not items:
            return
        table = items[0].table()
        cb = inserter(
            self.get_connection(), 
            table=table,
            policy=policy,
        )

        cb(with_items(*items))

    def add_hosts(self, hosts: pd.DataFrame) -> None:
        cb = inserter(self.get_connection(), table="hosts")
        cb(hosts)

    def with_view(self, name: str, q: str) -> "Repository":
        c = copy.copy(self)
        con = c.get_connection()
        con.sql(f"CREATE OR REPLACE VIEW {name} AS {q}")
        c._collection = con.sql(f"SELECT * FROM {name}")
        return c

    def collect(self) -> pd.DataFrame:
        return self._collection.df()

    def query(self, q: str, bsize=DEFAULT_BSIZE) -> Generator[Any, None, None]:
        cursor = self.get_connection().execute(q)
        cols = [c[0] for c in cursor.description]  # type: ignore

        while True:
            if rows := cursor.fetchmany(bsize):
                for r in rows:
                    yield dict(zip(cols, r))
                continue
            break

    def query_batch(
        self, q, normalize: bool = True, bsize=DEFAULT_BSIZE
    ) -> Generator[pd.DataFrame, None, None]:
        con = self.connector.new_connection()
        cursor = con.execute(q)
        norm = normalize_records if normalize else lambda x: x

        for b in cursor.fetch_record_batch(bsize):
            yield norm(b.to_pandas())

    def queryb(
        self, q, normalize: bool = True, bsize=DEFAULT_BSIZE
    ) -> tuple[int, Generator[Any, None, None]]:
        dq = f"WITH ct AS ({q}) SELECT COUNT(*) AS rows FROM ct;"
        con = self.connector.new_connection()
        d = (
            res[0] if (res := con.execute(dq).fetchone()) else 0
        )  # <---- this destroys previous queries, so we need a new connection for it
        return d, self.query_batch(q, normalize, bsize)
    

class Monitor:
    repo: Repository

    def __init__(self,
        on_init: list[HealthCheck],
        on_synchronize: list[HealthCheck],
    ) -> None:
        self._on_init=on_init
        self._on_synchronize=on_synchronize

    def initialize(self, repo: Repository): 
        self.repo = repo
        self.load(new_event(E_LOAD))

    def load(self, e: Event):
        print("running initialization checks")
        for check in self._on_init:
            check(self.repo, e)

    def synchronize(self, e: Event):
        print("running synchronization checks")
        for check in self._on_synchronize:
            check(self.repo, e)

    def sanity_check(self):
        print("checking repo health")
        e = new_event(E_SANITY)
        self.synchronize(e)

# TODO: the event has the info about the tables to rebuild, may be wise to rebuild only those
def rebuild_views(repo: Repository, _) -> None:
    print("rebuilding views...")

    tables = (
        repo.get_connection()
        .execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_name LIKE 'records_%'
            AND table_type = 'BASE TABLE'
        ORDER BY table_name
    """)
        .fetchall()
    )

    grouped = defaultdict(list)
    for (tname,) in tables:
        prefix = "_".join(tname.split("_")[:-1])
        grouped[prefix].append(tname)

    for prefix, tnames in grouped.items():
        union_sql = " UNION ALL ".join([f'SELECT * FROM "{t}"' for t in tnames])
        try:
            repo.get_connection().execute(
                f'CREATE OR REPLACE VIEW "{prefix}" AS {union_sql}'
            )
        except Exception as e:
            print(f"failed to rebuild records view: {prefix}")
            raise e
        
def get_records_views(repo: Repository) -> list[tuple[str]]:
    return (
        repo.get_connection()
        .execute("""
            SELECT table_name
            FROM information_schema.views
            WHERE table_name LIKE 'records_%'
        """)
        .fetchall()
    )
        
def find_host_col(repo: Repository, table: str) -> str | None:
    guesswork = {"ip", "saddr", "host", "addr"}
    q = f"SELECT * FROM '{table}' LIMIT 0"
    cur = repo.get_connection().execute(q)
    cols = {desc[0].lower() for desc in cur.description}

    common = guesswork & cols
    return next(iter(common), None)


def add_hosts_from_source(repo: Repository, db: str, col: Optional[str] = None) -> None:
    if not db.startswith("records_"):
        db = f"'records_{db}'"

    if not col:
       col = find_host_col(repo, db)
       if not col:
           print(f"fialed to find a host column in {db}")
           return
       
    # Check if hosts table exists
    con = repo.get_connection()
    if table_exists(con, "hosts"):
        q = f"""
        SELECT DISTINCT s.{col} AS ip
        FROM {db} AS s
        ANTI JOIN hosts AS h
          ON s.{col} = h.ip
        """
    else:
        q = f"""
        SELECT DISTINCT s.{col} AS ip
        FROM {db} AS s
        """

    t, gen = repo.queryb(q)
    if t == 0:
        print(f"no missing hosts from {db}")
        return
    
    with tqdm(total=t, desc="Hosts") as pbar:
        pbar.write("inserting missing hosts")
        for b in gen:
            hosts = [new_host(ip=r.ip) for r in b.itertuples()]
            repo.create(*hosts)
            pbar.update(len(b))
        
def add_missing_hosts(repo: Repository, e: Event):
    print("adding missing hosts...")

    if "table" not in e.summary:
        print("adding hosts from all views...")
        tabs = get_records_views(repo)
        for (tab,) in tabs:
            add_hosts_from_source(repo, tab)
        return
    
    table = e.summary["table"]
    add_hosts_from_source(repo, table)

def create_tables(models: list[Model]) -> HealthCheck:
    print(f"creating required tables: {','.join([m.table() for m in models])}")
    # create a fake module
    def hc(repo: Repository, _):
        con = repo.get_connection()
        for model in models:
            table = model.table()
            ins = inserter(
                con,
                table,
                pkey=model.primary_key(),
                policy=OnConflict.IGNORE
            )
            ins(model.mock())
    return hc

def new_repository(connector: Connector, monitor: Monitor) -> Repository:
    return Repository(
        connector=connector,
        monitor=monitor,
    )

def load_repository(
    sources: list[Source] = [], 
    db: str | None = None, 
    read_only: bool = False,
    save: str | None = None,
) -> Repository:
    connector = Connector(db, read_only=read_only)
    monitor = Monitor(
        on_init=[create_tables(M_REQUIRED)],
        on_synchronize=[
            rebuild_views,
            add_missing_hosts,
        ]
    )
    repo = new_repository(connector, monitor)
    repo.set_sources_path(save)
    repo.add_sources(*sources)
    return repo
