from copy import copy
from typing import Optional
from sqlalchemy import MetaData
from typing_extensions import Annotated

from analysis.tools import new_anonymizer, new_remover
from dice.start import load_repository
from dice.ast import make_parser
from dice.info import new_info

import ujson
import typer
import pandas as pd

search_app = typer.Typer(help="Query the database")

def normalize_services(services):
    if not services:
        return []

    # SQLite often returns stringified JSON
    if isinstance(services, str):
        try:
            services = ujson.loads(services)
        except Exception:
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
            except Exception:
                pass
        out.append(s)

    return out

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    cp = copy(df)
    if "services" in cp.columns:
        cp["services"] = cp["services"].apply(normalize_services)
    return cp

def anonymize_col(df: pd.DataFrame, col: str):
    mapping = {
        v: i
        for i, v in enumerate(df[col].unique(), start=1)
    }

    df[col] = df[col].map(mapping)

@search_app.command()
def search(
    q: str = typer.Option(
        "",
        "-q",
        "--query",
    ), 
    database: Optional[str] = typer.Option(
        None,
        "-db",
        "--database",
    ), 
    anonymize: Annotated[str, typer.Option()] = "",
    remove: Annotated[str, typer.Option()] = "",
    fields: Annotated[str, typer.Option()] = "hosts,ports,services,labels,tags", 
    exclude: Annotated[str, typer.Option()] = "",
) -> None:
    parser = make_parser()
    qt = parser.to_sql(q)

    repo = load_repository(db=database)
    n, batches = repo.query(qt)
    
    print(f"found {n} hosts")
    if not n:
        return

    flist = fields.split(",")
    if exclude:
        flist = list(set(flist) - set(exclude.split(",")))

    clist = anonymize.split(",")
    anzr = new_anonymizer(clist)

    rlist = remove.split(",")
    rm = new_remover(rlist)

    proc = (
        normalize, 
        rm.remove, 
        anzr.anonymize
    )

    with repo.connect() as con:
        info_b = new_info(flist)
        meta = MetaData()
        meta.reflect(bind=con)

        for b in batches:
            ips = b.ip.tolist()
            iq = info_b.make(ips, meta.tables)

            rows = con.execute(iq).mappings().all()
            df = pd.DataFrame(rows)

            for p in proc:
                df = p(df)

            print(df.to_json(orient="records", lines=True, force_ascii=False))

