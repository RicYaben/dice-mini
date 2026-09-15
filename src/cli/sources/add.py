from pathlib import Path
from typing import Annotated

from cyclopts import Parameter

from dice.cli.config.args import BatchSizeArg, ResultsArg
from dice.results import results, source


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
    rdb: ResultsArg | None = None,
    bsize: BatchSizeArg = 50_000,
):
    if not name:
        name = fpath.stem

    res = results(rdb)
    src = source(res, name)
    src.add(fpath, bsize, resume)
