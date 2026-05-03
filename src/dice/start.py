
from sqlalchemy import select

from dice.middlewares import add_missing_hosts
from dice.repo import Repository, new_repository
from dice.models import Cursor, Resource
from dice.database import new_connector
from dice.config import DEFAULT_BSIZE
from dice.health import HealthCheck, new_health_monitor
from dice.resources import new_resourcerer

def resume_cursors(repo: Repository) -> HealthCheck:
    def hc(_):
        with repo.connect() as con:
            stmt = (
                select(Resource)
                .join(Cursor)  # or .join(Resource.cursor)
                .where(Cursor.index != -1)
            )
            
            rsrcs = con.execute(stmt).fetchall()
            for res, in rsrcs:
                r = new_resourcerer(res, True, DEFAULT_BSIZE)
                r.cast(con)
    return hc


def load_repository(
    db: str | None = None,
) -> Repository:
    connector = new_connector(db)
    repo = new_repository(connector)

    init_hc = [resume_cursors(repo)]
    sync_hc = [add_missing_hosts(repo)]

    monitor = new_health_monitor(init_hc, sync_hc)
    return repo.load(monitor)