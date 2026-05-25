from pathlib import Path

import typer

from dice.internal.database import get_or_create
from dice.internal.loaders import walk
from dice.internal.resources import add_resource
from dice.internal.config import DEFAULT_BSIZE
from dice.shared.models import Source
from dice.cli.tools import load_repository

source_app = typer.Typer(help="Insert a source into the database")

@source_app.command()
def add(
    name: str = typer.Argument(
        help="Name of the source, e.g., zgrab2, zmap, etc."
    ),
    fpath: str | None = typer.Option(
        None,
        "-f",
        "--fpath",
        help="Source filepath"
    ),
    database: str | None = typer.Option(
        None, "-db", "--database", help="path to database"
    ),
    batch: int = typer.Option(
        DEFAULT_BSIZE,
        "-b",
        "--batch",
        help="batch size to read from each source. default 50K",
    ),
    resume: bool = typer.Option(
        True,
        "--resume"
    )
):
    repo = load_repository(db=database)
    with repo.session() as s:
        src, _ = get_or_create(s, Source, name=name)

    if not fpath:
        fpath = name
        if not Path(name).is_dir():
            fpath += ".*"

    for p in walk(fpath):
        add_resource(repo, src, str(p), resume=resume, bsize=batch) # type: ignore
