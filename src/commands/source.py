import typer

from dice.database import get_or_create
from dice.loaders import walk
from dice.models import Source
from dice.resources import add_resource
from dice.repo import load_repository
from dice.config import DEFAULT_BSIZE

source_app = typer.Typer(help="Insert a source into the database")

@source_app.command()
def add(
    name: str = typer.Argument(
        help="Name of the source, e.g., zgrab2, zmap, etc."
    ),
    fpath: str = typer.Argument(
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

    for p in walk(fpath):
        add_resource(repo, name, src.id.hex, str(p), resume=resume, bsize=batch)
