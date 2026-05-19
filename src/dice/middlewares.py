import logging

from sqlalchemy import inspect
from tqdm import tqdm
from sqlmodel import exists, select, text
from typing import Optional

from dice.constructors import new_host
from dice.database import insert_or_ignore
from dice.events import Event
from dice.models import Host, get_records_table
from dice.repo import Repository, query_batch, query_count
from dice.health import HealthCheck

logger = logging.getLogger(__name__)


def find_host_col(repo: Repository, table: str) -> str | None:
    guesswork = {"ip", "saddr", "host", "addr"}
    q = f"SELECT * FROM '{table}' LIMIT 0"

    with repo.connect() as con:
        res = con.execute(text(q))
        cols = {desc[0].lower() for desc in res.cursor.description}  # type: ignore

        common = guesswork & cols
        return next(iter(common), None)


def add_hosts_from_records_table(
    repo: Repository, name: str, col: Optional[str] = "ip"
) -> None:
    if not col:
        col = find_host_col(repo, name)
    if not col:
        logger.debug(f"fialed to find a host column in {name}")
        return

    with repo.connect() as con:
        tab = get_records_table(con, name)
        c = getattr(tab.c, col)

        q = str(
            select(c.distinct().label("ip"))
            .where(~exists().where(c == Host.ip))
            .compile(con)
        )

        n = query_count(q, con)
        if not n:
            logger.debug(f"no missing hosts from {name}")
            return

    with tqdm(total=n, desc="Hosts") as pbar:
        pbar.write("inserting missing hosts")

        with repo.session() as ses:
            for b in query_batch(q, ses.connection()):
                hosts = [new_host(ip=str(r.ip)) for r in b]
                inserted = insert_or_ignore(ses, Host, hosts)
                pbar.update(len(b))


def add_missing_hosts(repo: Repository) -> HealthCheck:
    def hc(e: Event):
        if "table" not in e.summary:
            logger.debug("adding hosts from all views...")
            with repo.session() as s:
                insp = inspect(s.get_bind())
                tabs = [
                    name for name in insp.get_table_names() if name.endswith("_records")
                ]

            for tab in tabs:
                add_hosts_from_records_table(repo, tab)
            return

        logger.debug("adding missing hosts...")
        table = e.summary["table"]
        add_hosts_from_records_table(repo, table)

    return hc
