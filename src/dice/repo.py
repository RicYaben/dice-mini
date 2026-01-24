from collections import defaultdict
from typing import Any, Generator, Callable, Optional
from tqdm import tqdm

from dice.connector import Connector
from dice.health import HealthCheck, HealthMonitor, new_health_monitor
from dice.models import Model, Source, M_REQUIRED
from dice.helpers import new_collection, new_host
from dice.config import DEFAULT_BSIZE, DATA_PREFIX
from dice.store import OnConflict, inserter, store, table_exists
from dice.events import Event, new_event, EventType
from dice.watcher import Watcher, new_watcher

import pandas as pd
import duckdb
import ujson
import warnings
import logging

warnings.simplefilter(action="ignore", category=UserWarning)

logger = logging.getLogger(__name__)

type RecordsWrapper = Callable[[Any], pd.DataFrame]

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


class Repository:
    _collection: duckdb.DuckDBPyRelation

    def __init__(
        self, 
        connector: Connector,
    ) -> None:
        self.connector = connector

    def load(self, monitor: HealthMonitor) -> 'Repository':
        self.monitor = monitor
        monitor.initialize()
        return self

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

    def add_sources(self, 
        sources: list[Source], 
        resume: bool = True
    ):
        for src in sources:
            self.add_source(src, resume)

    def add_source(self, src: Source, resume: bool = True):
        logger.info(f"adding {src.name} records ({src.batch_size}/b)")

        # Try to insert the source
        self.create(src)

        # watch over the source, we want to track the cursor in case of failure
        # to resume ading the source
        guard = self.watch(src, resume)
        con = self.get_connection()

        insert_conf={
            "con": con, 
            "table": guard.table, 
            "policy": OnConflict.FORCE
        }

        store_conf={
            "dir": self._sources_path, 
            "fname": src.name
        }

        with store(store_conf, insert_conf) as s:
            gen = guard.consume()
            for c in tqdm(gen, desc=src.name):
                s.save(c)
                del c

        summary = {
            "table": guard.table,
            "iconf": insert_conf,
            "sconf": store_conf,
        }
        event = new_event(EventType.SOURCE, summary)
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

    def watch(self, src: Source, resume: bool) -> Watcher:
        # TODO: support non-local records?
        table = make_table_name(src.name, src.digest)
        cursor = make_cursor(self, table, resume)
        watcher = new_watcher(table, src, cursor)

    def simple_query(self, q: str, bsize: int= DEFAULT_BSIZE) -> Generator[dict, None, None]:
        """Execute a query in batches. Returns a dict

        Args:
            q (str): query
            bsize (int, optional): Batch size. Defaults to DEFAULT_BSIZE.

        Yields:
            Generator[dict, None, None]: Dataset (chunked)
        """
        cursor = self.get_connection().execute(q)
        cols = [c[0] for c in cursor.description]  # type: ignore

        while True:
            if rows := cursor.fetchmany(bsize):
                for r in rows:
                    yield dict(zip(cols, r))
                continue
            break

    def query_batch(
        self, q: str, normalize: bool = True, bsize: int =DEFAULT_BSIZE
    ) -> Generator[pd.DataFrame, None, None]:
        """Execute a query in batches. Returns a pandas dataframe

        Args:
            q (str): query
            normalize (bool, optional): Wether to normalize records. Defaults to True.
            bsize (int, optional): Batch size. Defaults to DEFAULT_BSIZE.

        Yields:
            Generator[pd.DataFrame, None, None]: Dataset (chunked)
        """
        con = self.connector.new_connection()
        cursor = con.execute(q)
        norm = normalize_records if normalize else lambda x: x

        for b in cursor.fetch_record_batch(bsize):
            yield norm(b.to_pandas())

    def query_count(self, q: str) -> int:
        """Count number of items in the query

        Args:
            q (str): query

        Returns:
            int: number of results
        """
        dq = f"WITH ct AS ({q}) SELECT COUNT(*) AS rows FROM ct;"
        con = self.connector.new_connection()
        d = (
            res[0] if (res := con.execute(dq).fetchone()) else 0
        )  # <---- this destroys previous queries, so we need a new connection for it
        return d

    def query(
        self, q: str, normalize: bool = True, bsize: int=DEFAULT_BSIZE
    ) -> tuple[int, Generator[pd.DataFrame, None, None]]:
        """A wrapper for the query to return the number of results in the query and the batches

        Args:
            q (str): query
            normalize (bool, optional): Wether to normalize records. Defaults to True.
            bsize (int, optional): size of the batch. Defaults to DEFAULT_BSIZE.

        Returns:
            tuple[int, Generator[pd.DataFrame, None, None]]: number of results, and dataset
        """
        d = self.query_count(q)
        gen = self.query_batch(q, normalize, bsize)
        return (d, gen)
        
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


def add_hosts_from_records_table(repo: Repository, db: str, col: Optional[str] = None) -> None:
    if not db.startswith("records_"):
        db = f"'records_{db}'"

    if not col:
       col = find_host_col(repo, db)
       if not col:
           logger.info(f"fialed to find a host column in {db}")
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

    t, gen = repo.query(q)
    if t == 0:
        logger.info(f"no missing hosts from {db}")
        return
    
    with tqdm(total=t, desc="Hosts") as pbar:
        pbar.write("inserting missing hosts")
        for b in gen:
            hosts = [new_host(ip=str(r.ip)) for r in b.itertuples()]
            repo.create(*hosts)
            pbar.update(len(b))

def add_missing_hosts(repo: Repository) -> HealthCheck:
    def hc(e: Event):
        if "table" not in e.summary:
            logger.info("adding hosts from all views...")
            tabs = get_records_views(repo)
            for (tab,) in tabs:
                add_hosts_from_records_table(repo, tab)
            return
        
        logger.info("adding missing hosts...")
        table = e.summary["table"]
        add_hosts_from_records_table(repo, table)
    return hc

# TODO: the event has the info about the tables to rebuild, may be wise to rebuild only those
def rebuild_views(repo: Repository) -> HealthCheck:
    def hc(_) -> None:
        logger.info("rebuilding views...")

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
                logger.info(f"failed to rebuild records view: {prefix}")
                raise e
    return hc

def create_tables(repo: Repository, models: list[Model]) -> HealthCheck:
    logger.info(f"creating required tables: {','.join([m.table() for m in models])}")
    # create a fake module
    def hc(_):
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

def resume_cursors(repo: Repository) -> HealthCheck:
    def hc(_):
        raise NotImplementedError
    return hc

def new_repository(connector: Connector) -> Repository:
    return Repository(
        connector=connector,
    )

def load_repository(
    db: str | None = None, 
    read_only: bool = False,
    save: str | None = None,
) -> Repository:
    connector = Connector(db, read_only=read_only)
    repo = new_repository(connector)

    init_hc = [
            create_tables(repo, M_REQUIRED),
            resume_cursors(repo), # something happen before, resume pending cursors
        ]
    
    sync_hc = [
        rebuild_views(repo),
        add_missing_hosts(repo)
    ]

    monitor = new_health_monitor(init_hc, sync_hc)
    repo.set_sources_path(save)
    return repo.load(monitor)
