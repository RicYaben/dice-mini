from dice.cli.config.args import ResultsArg
from dice.results import results


def health(rdb: ResultsArg | None = None):
    repo = results(rdb)
    repo.synchronize()
