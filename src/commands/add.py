import typer

from dice.repo import load_repository
from dice.helpers import new_source
from dice.config import DEFAULT_BSIZE

insert_app = typer.Typer(help="Insert a source into the database")

@insert_app.command()
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
    src = new_source(name, fpath, "-", batch_size=batch)
    repo.add_source(src, resume)
