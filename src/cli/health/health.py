from dice.cli.tools import load_repository


def health(
    database: str
):
    repo = load_repository(db=database)
    repo.monitor.sanity_check()
