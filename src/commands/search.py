from typing import Annotated

import pandas as pd
import typer
import ujson

from analysis.tools import new_anonymizer, new_remover
from dice.cli.tools import load_repository
from dice.internal.ast import make_parser
from dice.internal.info import new_info
from dice.shared.query import to_sql

search_app = typer.Typer(help="Query the database")


def normalize_services(services):
    if not services:
        return []

    # SQLite often returns stringified JSON
    if isinstance(services, str):
        try:
            services = ujson.loads(services)
        except Exception as e:
            print(e)  # Not going to handle this?
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
                s["data"] = ujson.loads(s["data"])
            except Exception as e:
                print(e)  # Not going to handle this?
        out.append(s)

    return out


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    if "services" in df.columns:
        df["services"] = df["services"].apply(normalize_services)
    return df


def anonymize_col(df: pd.DataFrame, col: str):
    mapping = {v: i for i, v in enumerate(df[col].unique(), start=1)}
    df[col] = df[col].map(mapping)


@search_app.command()
def search(
    q: str = typer.Option(
        "",
        "-q",
        "--query",
    ),
    database: str | None = typer.Option(
        None,
        "-db",
        "--database",
    ),
    limit: int | None = typer.Option(None, "-l", "--limit"),
    # TODO: store mappings
    anonymize: Annotated[str, typer.Option()] = "",
    mappings: Annotated[str | None, typer.Option()] = None,
    remove: Annotated[str, typer.Option()] = "",
    fields: Annotated[str, typer.Option()] = "ports,services,labels,tags",
    exclude: Annotated[str, typer.Option()] = "",
) -> None:
    flist = fields.split(",")
    parser = make_parser()
    qt = parser.to_sql(q)

    repo = load_repository(db=database)
    res = repo.search(qt, limit=limit)
    n = res.count()

    print(f"found {n} hosts")
    if not n:
        return

    procs = [normalize]
    if remove:
        rlist = remove.split(",")
        rm = new_remover(rlist)
        procs.append(rm.remove)

    if anonymize:
        clist = anonymize.split(",")
        anzr = new_anonymizer(clist, mappings)
        procs.append(anzr.anonymize)

    if exclude:
        flist = list(set(flist) - set(exclude.split(",")))

    info_b = new_info(flist)
    for b in res.df(50_000):
        ips = b["ip"].tolist()  # type: ignore
        qs = info_b.make(ips)
        rows = repo.search(to_sql(qs)).all()
        df = pd.DataFrame(rows)

        for p in procs:
            df = p(df)

        print(df.to_json(orient="records", lines=True, force_ascii=False))
