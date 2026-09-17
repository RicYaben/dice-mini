from pathlib import Path

from dice.internal.database import new_connector
from dice.internal.monitor.health import new_health_monitor
from dice.internal.monitor.middlewares import add_missing_hosts, resume_cursors
from dice.internal.results.repository import new_repository
from dice.shared.interfaces import Repository
from dice.shared.models import DatabaseModel


def results(res: str | Path | None) -> Repository:
    connector = new_connector(res, DatabaseModel)
    repo = new_repository(connector)

    init_hc = [resume_cursors(repo)]
    sync_hc = [add_missing_hosts(repo)]

    monitor = new_health_monitor(init_hc, sync_hc)
    return repo.load(monitor)
