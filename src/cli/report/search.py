from logging import getLogger

import pandas as pd
import ujson

from analysis.tools import new_anonymizer, new_remover
from dice.cli.config.args import BatchSizeArg, ResultsArg
from dice.cli.tools import load_repository
from dice.internal.ast import make_parser
from dice.internal.report import ReportBuilder, ReportOptions
from dice.shared.query import to_sql

from .context import SearchOptions, make_context

logger = getLogger(__name__)


class UnmarshallError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


def normalize_services(services):
    if not services:
        return []

    # SQLite often returns stringified JSON
    if isinstance(services, str):
        try:
            services = ujson.loads(services)
        except ujson.JSONDecodeError as e:
            logger.warning(f"Failed to unmarshal services: {e}")
            return []

    # single object → list
    if isinstance(services, dict):
        services = [services]

    # final safe transform
    out = []
    for s in services:
        if not isinstance(s, dict):
            continue
        s = dict(s)
        if "data" in s and isinstance(s["data"], str):
            try:
                d = ujson.loads(s["data"])
                s["data"] = d
            except ujson.JSONDecodeError as e:
                logger.warning(f"Failed to unmarshal data object: {e}")
        out.append(s)

    return out


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    if "services" in df.columns:
        df["services"] = df["services"].apply(normalize_services)
    return df


def search(
    opts: SearchOptions | None = None,
    results: ResultsArg | None = None,
    bsize: BatchSizeArg = 50_000,
) -> None:
    ctx = make_context(opts)

    qt = make_parser().to_sql(ctx.search.query or "")

    repo = load_repository(db=results)
    res = repo.search(
        qt, limit=ctx.search.limit
    )  # TODO: add offset (pagination support)

    n = res.count()
    if n == 0:
        print("Query returned no results: ", ctx.search.query or "<empty>")
        return
    print(f"Found {n} hosts")

    procs = [normalize]
    if rfields := ctx.anonimizer.remove:
        rm = new_remover(rfields)
        procs.append(rm.remove)

    if afields := ctx.anonimizer.anonimize:
        anzr = new_anonymizer(afields, ctx.anonimizer.output)
        procs.append(anzr.anonymize)

    flist = ctx.search.include or []
    if exclude := ctx.search.exclude:
        flist = list(set(flist) - set(exclude))

    options = ReportOptions.from_fields(flist)
    rbuilder = ReportBuilder(options)

    for batch in res.batch(bsize):
        ips = [row.ip for row in batch]
        qs = rbuilder.build(ips)
        rows = repo.search(to_sql(qs)).df()

        assert isinstance(rows, pd.DataFrame)
        for proc in procs:
            rows = proc(rows)
        print(rows)
