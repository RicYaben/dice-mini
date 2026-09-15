from typing import Annotated

from cyclopts import Parameter

# TODO: move analysis package to shared reports
from analysis.comparing import compare
from dice.cli.config.models import SearchOptions
from dice.results import results


def diff(
    left: Annotated[
        str, Parameter(name=["--left", "-l"], help="Path to the first dice database.")
    ],
    right: Annotated[
        str, Parameter(name=["--right", "-r"], help="Path to the second dice database.")
    ],
    opts: SearchOptions | None = None,
) -> None:
    if opts is None:
        opts = SearchOptions()

    flist = opts.include or []
    if exclude := opts.exclude:
        flist = list(set(flist) - set(exclude))

    r1 = results(left)
    r2 = results(right)

    for result in compare(r1, r2, opts.query, flist):
        print(result)
