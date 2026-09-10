from pathlib import Path
from typing import Annotated

from cyclopts import Parameter

from dice.cli.config.args import BatchSizeArg, ResultsArg
from dice.cli.tools import load_repository
from dice.internal.database import get_or_create
from dice.internal.loaders import walk
from dice.internal.resources import add_resource
from dice.shared.models import Source


def add(
    fpath: Annotated[
        Path,
        Parameter(name=["--fpath", "-f"], help="Path to the source file or directory"),
    ],
    name: Annotated[
        str | None, Parameter(name=["--name", "-n"], help="Name of the source")
    ] = None,
    resume: Annotated[
        bool, Parameter(name=["--resume", "-r"], help="Resume adding a source")
    ] = True,
    results: ResultsArg | None = None,
    bsize: BatchSizeArg = 50_000,
):
    if not name:
        name = fpath.stem

    repo = load_repository(db=results)
    with repo.session() as s:
        src, _ = get_or_create(s, Source, name=name)

    for p in walk(fpath):
        add_resource(repo, src, p, resume=resume, bsize=bsize)
