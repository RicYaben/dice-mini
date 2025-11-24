import os
from typing import List
import typer
from dice.repo import load_repository
from dice.helpers import make_sources

insert_app = typer.Typer(help="Insert a source into the database")

def parse_source(entry: str):
    """
    Split entry into 1–3 fields.
    Example inputs:
        "a.jsonl"
        "a.jsonl zmap"
        "a.jsonl zmap study1"
    """
    parts = entry.split()

    if len(parts) == 0:
        raise Exception("Empty -s entry")

    if len(parts) > 3:
        raise Exception(f"Too many fields in -s entry: {entry}")

    fpath = parts[0]

    # Default: name = filename without extension
    if len(parts) >= 2:
        name = parts[1]
    else:
        name = os.path.splitext(os.path.basename(fpath))[0]

    # Default: study = "-"
    if len(parts) >= 3:
        study = parts[2]
    else:
        study = "-"

    return (fpath, name, study)

@insert_app.command()
def add(
    source: List[str] = typer.Option(
        [], 
        "-s", 
        "--source", 
        help="source path and name"
    ),
    database: str | None = typer.Option(
        None, 
        "-db",
        "--database",
        help="path to database"
    ),
    batch: int = typer.Option(
        10_000,
        "-b",
        "--batch",
        help="batch size to read from each source. default 10K"
    )
):
    repo = load_repository(db=database)
    if not source:
        print("no source to insert")
        raise typer.Abort()
    
    srcs = []
    for s in source:
        spath, sname, study = parse_source(s)
        srcs = make_sources(spath, name=sname, study=study, batch_size=batch) 
    repo.add_sources(*srcs)