from collections.abc import Sequence
from typing import Annotated

import ujson
from cyclopts import Parameter

from dice.cli.args import BatchSizeArg
from dice.cli.models import SearchOptions
from dice.reports import compare
from dice.results import results


def compare_cmd(
    repos: Annotated[
        Sequence[str],
        Parameter(name=["--results", "-r"], help="Path to DICE results."),
    ],
    opts: SearchOptions | None = None,
    bsize: BatchSizeArg | None = None,
) -> None:
    if opts is None:
        opts = SearchOptions()

    flist = opts.include or []
    if exclude := opts.exclude:
        flist = list(set(flist) - set(exclude))

    res = [results(r) for r in repos]
    for comp in compare(res, flist, opts.query, bsize):
        print(ujson.dumps(comp.to_dict()))
