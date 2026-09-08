from analysis.comparing import compare
from dice.cli.tools import writer
from dice.internal.database import new_connector
from dice.internal.repository import new_repository


def diff(
    q: str,
    d1: str,
    d2: str,
    fields: str = "hosts,ports,services",
    exclude: str = "",
    output: str = "",
) -> None:
    flist = fields.split(",")
    if exclude:
        flist = list(set(flist) - set(exclude.split(",")))

    r1 = new_repository(new_connector(d1))
    r2 = new_repository(new_connector(d2))

    with writer(output) as w:
        compare(r1, r2, q, flist, w)
