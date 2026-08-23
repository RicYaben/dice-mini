from typing import Annotated

import typer

from analysis.comparing import compare
from dice.internal.database import new_connector
from dice.internal.repository import new_repository

compare_app = typer.Typer(help="Compare two datasets")


@compare_app.command(name="compare")
def diff(
    q: str,
    d1: str = typer.Argument(help="Base dataset"),
    d2: str = typer.Argument(help="Comparing dataset"),
    fields: Annotated[str, typer.Option()] = "hosts,ports,services",
    exclude: Annotated[str, typer.Option()] = "",
    output: Annotated[str, typer.Option()] = "comparison.jsonl",
) -> None:
    flist = fields.split(",")
    if exclude:
        flist = list(set(flist) - set(exclude.split(",")))

    r1 = new_repository(new_connector(d1))
    r2 = new_repository(new_connector(d2))
    compare(r1, r2, q, flist, output)
