from logging import getLogger

# TODO: move analysis tools to shared `reports` package
from analysis.tools import new_anonymizer, new_remover
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

    procs = []
    if rfields := ctx.anonimizer.remove:
        rm = new_remover(rfields)
        procs.append(rm.remove)

    if afields := ctx.anonimizer.anonimize:
        anzr = new_anonymizer(afields, ctx.anonimizer.output)
        procs.append(anzr.anonymize)

    flist = ctx.search.include or []
    if exclude := ctx.search.exclude:
        flist = list(set(flist) - set(exclude))

    for r in reports(repo, ctx.search.query, flist, bsize):
        r = (proc(r) for proc in procs)
