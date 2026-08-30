import sys
from contextlib import nullcontext

from dice.internal.cookbook import Cookbook, new_cookbook
from dice.internal.database import new_connector
from dice.internal.health import new_health_monitor
from dice.internal.middlewares import add_missing_hosts, resume_cursors
from dice.internal.models import CookbookModel
from dice.internal.repository import Repository, new_repository
from dice.shared.models import DatabaseModel


def load_repository(
    db: str | None = None,
) -> Repository:
    connector = new_connector(db, DatabaseModel)
    repo = new_repository(connector)

    init_hc = [resume_cursors(repo)]
    sync_hc = [add_missing_hosts(repo)]

    monitor = new_health_monitor(init_hc, sync_hc)
    return repo.load(monitor)


def load_cookbook(db: str | None = None) -> Cookbook:
    con = new_connector(db, CookbookModel)
    repo = new_repository(con)
    cb = new_cookbook(repo)
    return cb


def writer_ctx(output: str | None) -> object:
    if not output:
        return nullcontext(sys.stdout)
    return open(output, "+a")

class Writer:
    def __init__(self, output: str | None):
        self._ctx = writer_ctx(output)

    def __enter__(self):
        return self._ctx.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._ctx.__exit__(exc_type, exc_val, exc_tb)

def writer(output: str | None) -> Writer:
    return Writer(output)
