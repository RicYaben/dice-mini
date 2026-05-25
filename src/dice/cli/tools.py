from dice.internal.middlewares import add_missing_hosts, resume_cursors
from dice.internal.repository import Repository, new_repository
from dice.internal.database import new_connector
from dice.internal.health import new_health_monitor


def load_repository(
    db: str | None = None,
) -> Repository:
    connector = new_connector(db)
    repo = new_repository(connector)

    init_hc = [resume_cursors(repo)]
    sync_hc = [add_missing_hosts(repo)]

    monitor = new_health_monitor(init_hc, sync_hc)
    return repo.load(monitor)
