import logging

from sqlmodel import exists, select
from tqdm import tqdm

from dice.shared.models import Cursor, Host, Record, Resource

from .config import DEFAULT_BSIZE
from .database import insert_or_ignore
from .events import Event
from .health import HealthCheck
from .repository import Repository, query_batch, query_count
from .resources import new_resourcerer

logger = logging.getLogger(__name__)


def add_hosts_from_records_table(repo: Repository) -> None:
    with repo.connect() as con:
        q = str(
            select(Record.host.distinct().label("ip"))  # type: ignore
            .where(~exists().where(Record.host == Host.ip))  # type: ignore
            .compile(con)
        )

        n = query_count(q, con)
        if not n:
            logger.debug("no missing hosts from records")
            return

    with tqdm(total=n, desc="Hosts") as pbar:
        pbar.write("inserting missing hosts")

        with repo.session() as ses:
            for b in query_batch(q, ses.connection()):
                hosts = [Host(ip=str(r.ip)) for r in b]
                insert_or_ignore(ses, Host, hosts)
                pbar.update(len(b))


def add_missing_hosts(repo: Repository) -> HealthCheck:
    def hc(e: Event):
        logger.debug("adding missing hosts...")
        t = e.summary.get("table", None)
        if not t or t == Record.__tablename__:
            add_hosts_from_records_table(repo)

    return hc


def resume_cursors(repo: Repository) -> HealthCheck:
    def hc(_):
        con = repo.connect()
        stmt = select(Resource).join(Cursor).where(Cursor.idx != -1)

        rows = con.execute(stmt).all()
        for res in rows:
            r = new_resourcerer(res.id, True, DEFAULT_BSIZE)
            r.cast(con)

    return hc
