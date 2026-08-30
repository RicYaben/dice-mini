import typer

from dice.cli.tools import load_repository

health_app = typer.Typer(help="Run healthchecks on a database")


@health_app.command()
def health(
    database: str = typer.Argument(),
):
    repo = load_repository(db=database)
    repo.monitor.sanity_check()
