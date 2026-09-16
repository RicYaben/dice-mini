from typing import Annotated

from cyclopts import Parameter

from analysis.comparing import compare
from dice.internal.database import new_connector
from dice.internal.repository import new_repository


def diff(
    left: Annotated[
        str, Parameter(name=["--left", "-l"], help="Path to the first dice database.")
    ],
    right: Annotated[
        str, Parameter(name=["--right", "-r"], help="Path to the second dice database.")
    ],
    q: Annotated[
        str | None, Parameter(name=["--query", "-q"], help="query to execute.")
    ] = None,
    fields: Annotated[
        str,
        Parameter(
            name=["--fields", "-f"],
            help="Comma-separated list of fields to include in the comparison.",
        ),
    ] = "hosts,ports,services",
    exclude: Annotated[
        str | None,
        Parameter(
            name=["--exclude", "-e"],
            help="Comma-separated list of fields to exclude from the comparison.",
        ),
    ] = None,
) -> None:
    flist = fields.split(",")
    if exclude:
        flist = list(set(flist) - set(exclude.split(",")))

    r1 = new_repository(new_connector(left))
    r2 = new_repository(new_connector(right))

    for result in compare(r1, r2, q, flist):
        print(result)
