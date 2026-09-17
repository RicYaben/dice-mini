from dataclasses import asdict
from logging import getLogger

import ujson

from dice.cli.args import BatchSizeArg, ResultsArg
from dice.reports import reports
from dice.results import results

from .context import ReportOptions, make_context

logger = getLogger(__name__)


class UnmarshallError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


def report(
    opts: ReportOptions | None = None,
    fpath: ResultsArg | None = None,
    bsize: BatchSizeArg = 50_000,
) -> None:
    ctx = make_context(opts)
    repo = results(fpath)

    flist = ctx.search.include or []
    if exclude := ctx.search.exclude:
        flist = list(set(flist) - set(exclude))

    for r in reports(repo, ctx.search.query, flist, bsize):
        print(ujson.dumps(asdict(r)))
