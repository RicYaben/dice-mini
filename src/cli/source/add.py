from pathlib import Path

from dice.cli.tools import load_repository
from dice.internal.database import get_or_create
from dice.internal.loaders import walk
from dice.internal.resources import add_resource
from dice.shared.models import Source


def add(
    name: str,
    fpath: str | None = None,
    resume: bool = True,
    results: str | None = None,
    bsize: int = 50_000,
):
    repo = load_repository(db=results)
    with repo.session() as s:
        src, _ = get_or_create(s, Source, name=name)

    if not fpath:
        fpath = name
        if not Path(name).is_dir():
            fpath += ".*"

    for p in walk(fpath):
        add_resource(repo, src, str(p), resume=resume, bsize=bsize)
