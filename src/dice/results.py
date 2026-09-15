from pathlib import Path

from dice.internal.database import get, get_or_create, new_connector
from dice.internal.health import new_health_monitor
from dice.internal.loaders import walk
from dice.internal.middlewares import add_missing_hosts, resume_cursors
from dice.internal.repository import new_repository
from dice.internal.resources import add_resource
from dice.shared.interfaces import Repository
from dice.shared.models import DatabaseModel, Source


class SourceNotFoundError(Exception):
    def __init__(self, name: int) -> None:
        self.src_name = name
        super().__init__(f"source not found: {name}")


def results(res: str | Path | None) -> Repository:
    connector = new_connector(res, DatabaseModel)
    repo = new_repository(connector)

    init_hc = [resume_cursors(repo)]
    sync_hc = [add_missing_hosts(repo)]

    monitor = new_health_monitor(init_hc, sync_hc)
    return repo.load(monitor)


class SourceManager:
    def __init__(self, res: Repository, name: str, create: bool = True) -> None:
        self.repo = res
        with res.session() as s:
            if create:
                src, _ = get_or_create(s, Source, name=name)
            else:
                src = get(s, Source, name=name)

            if src is None:
                raise SourceNotFoundError(name)
            self.src = src

    def add(self, fpath: Path, bsize: int, resume: bool):
        for p in walk(fpath):
            add_resource(self.res, self.src, p, resume, bsize)


def source(res: Repository, name: str, create: bool = True) -> SourceManager:
    return SourceManager(res, name, create)
