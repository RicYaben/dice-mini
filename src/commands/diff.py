import typer
from typing_extensions import Annotated
from dice.repo import new_repository, new_connector
from dice.analysis.comparing import compare

diff_app = typer.Typer(help="Compare two datasets")

# NOTE: In principle this is NOT a function. It is supposed to be a module
@diff_app.command()
def diff(
    q: str,
    src: str = typer.Argument(help="Base dataset"),
    dst: str = typer.Argument(help="Comparing dataset"),
    fields: Annotated[str, typer.Option()] = "hosts,ports,services",
    exclude: Annotated[str, typer.Option()] = ""
) -> None:
    flist = fields.split(",")
    if exclude:
        flist = list(set(flist) - set(exclude.split(",")))

    connector = new_connector(src)
    connector.attach(dst, "dst")

    repo = new_repository(connector)
    compare(repo, q, "dst", flist)

