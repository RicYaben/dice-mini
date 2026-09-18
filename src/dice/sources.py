from pathlib import Path

from dice.internal.database import get, get_or_create
from dice.internal.sources.loaders import walk
from dice.internal.sources.resources import add_resource
from dice.shared.interfaces import Repository
from dice.shared.models import Source


class SourceNotFoundError(Exception):
    def __init__(self, name: str) -> None:
        self.src_name = name
        super().__init__(f"source not found: {name}")


class SourceManager:
    repo: Repository
    src: Source

    def __init__(self, res: Repository, name: str, create: bool = True) -> None:
        self.repo = res
        with res.session() as s:
            if create:
                src, _ = get_or_create(s, Source, name=name)
            else:
                src = get(s, Source, name=name)

            if src is None:
                raise SourceNotFoundError(name)

            assert isinstance(src, Source)
            self.src = src

    def add(self, fpath: Path, bsize: int, resume: bool):
        for p in walk(fpath):
            add_resource(self.repo, self.src, p, bsize, resume)


def source(res: Repository, name: str, create: bool = True) -> SourceManager:
    return SourceManager(res, name, create)
