from tqdm import tqdm
from typing import Any, Generator, Callable, Optional
from sqlmodel import Session, text
from sqlalchemy import Connection, inspect

from dice.health import HealthCheck, HealthMonitor, new_health_monitor
from dice.helpers import new_collection, new_host
from dice.config import DEFAULT_BSIZE, DATA_PREFIX
from dice.events import Event
from dice.database import Connector, insert_or_ignore, new_connector

import pandas as pd
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
    def __init__(
        self,
        con: Connector,
    ) -> None:
        self.con = con

    def load(self, monitor: HealthMonitor) -> "Repository":
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

        with self.connect() as conn:
            result = pd.read_sql(q, conn, params=params)
            return result

    def connect(self) -> Connection:
        return self.con.connection()

    def session(self) -> Session:
        return self.con.session()

    def insert(self, items: list[Any], policy=insert_or_ignore):
        if not items:
            return

        model = type(items[0])
        with self.session() as s:
            policy(s, model, items)

    def simple_query(
        self, q: str, bsize: int = DEFAULT_BSIZE
    ) -> Generator[dict, None, None]:
        """Execute a query in batches. Returns a dict

        Args:
            q (str): query
            bsize (int, optional): Batch size. Defaults to DEFAULT_BSIZE.

        Yields:
            Generator[dict, None, None]: Dataset (chunked)
        """
        with self.connect() as con:
            res = con.execute(text(q))
            cols = [c[0] for c in res.cursor.description]  # type: ignore

            while True:
                if rows := res.fetchmany(bsize):
                    for r in rows:
                        yield dict(zip(cols, r))
                    continue
                break

    def query_batch(
        self, q: str, normalize: bool = True, bsize: int = DEFAULT_BSIZE
    ) -> Generator[pd.DataFrame, None, None]:
        """Execute a query in batches. Returns a pandas dataframe

        Args:
            q (str): query
            normalize (bool, optional): Wether to normalize records. Defaults to True.
            bsize (int, optional): Batch size. Defaults to DEFAULT_BSIZE.

        Yields:
            Generator[pd.DataFrame, None, None]: Dataset (chunked)
        """
        with self.connect() as con:
            res = con.execute(text(q)).mappings()
            norm = normalize_records if normalize else lambda x: x

            while rows := res.fetchmany(bsize):
                yield norm(pd.DataFrame.from_records(rows)) # type: ignore

    def query_count(self, q: str) -> int:
        """Count number of items in the query

        Args:
            q (str): query

        Returns:
            int: number of results
        """
        dq = f"WITH ct AS ({q}) SELECT COUNT(*) AS rows FROM ct;"
        with self.connect() as con:
            d = (
                res[0] if (res := con.execute(text(dq)).fetchone()) else 0
            )  # <---- this destroys previous queries, so we need a new connection for it
            return d

    def query(
        self, q: str, normalize: bool = True, bsize: int = DEFAULT_BSIZE
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


def find_host_col(repo: Repository, table: str) -> str | None:
    guesswork = {"ip", "saddr", "host", "addr"}
    q = f"SELECT * FROM '{table}' LIMIT 0"

    with repo.connect() as con:
        res = con.execute(text(q))
        cols = {desc[0].lower() for desc in res.cursor.description}  # type: ignore

        common = guesswork & cols
        return next(iter(common), None)


def add_hosts_from_records_table(
    repo: Repository, table: str, col: Optional[str] = "ip"
) -> None:
    if not col:
        col = find_host_col(repo, table)
    if not col:
        logger.debug(f"fialed to find a host column in {table}")
        return

    # NOTE: we don't have a model for random records, so we have to deal with this
    q = f"""
    SELECT DISTINCT t.{col} AS ip
    FROM "{table}" AS t
    WHERE NOT EXISTS (
        SELECT NULL
        FROM hosts
        WHERE t.{col} = hosts.ip
    )
    """

    n, gen = repo.query(q)
    if not n:
        logger.debug(f"no missing hosts from {table}")
        return

    with tqdm(total=n, desc="Hosts") as pbar:
        pbar.write("inserting missing hosts")
        for b in gen:
            hosts = [new_host(ip=str(r.ip)) for r in b.itertuples()]
            repo.insert(hosts)
            pbar.update(len(b))

def add_missing_hosts(repo: Repository) -> HealthCheck:
    def hc(e: Event):
        if "table" not in e.summary:
            logger.debug("adding hosts from all views...")
            with repo.session() as s:
                insp = inspect(s.get_bind())
                tabs = [
                    name
                    for name in insp.get_table_names()
                    if name.endswith("_records")
                ]

            for (tab,) in tabs:
                add_hosts_from_records_table(repo, tab)
            return

        logger.debug("adding missing hosts...")
        table = e.summary["table"]
        add_hosts_from_records_table(repo, table)
    return hc


# TODO: implement this
def resume_cursors(repo: Repository) -> HealthCheck:
    def hc(_):
        raise NotImplementedError
    return hc


def new_repository(connector: Connector) -> Repository:
    return Repository(
        con=connector,
    )


def load_repository(
    db: str | None = None,
) -> Repository:
    connector = new_connector(db)
    repo = new_repository(connector)

    init_hc = []#resume_cursors(repo)]
    sync_hc = [add_missing_hosts(repo)]

    monitor = new_health_monitor(init_hc, sync_hc)
    return repo.load(monitor)
