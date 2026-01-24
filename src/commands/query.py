from typing import Optional
from typing_extensions import Annotated

from dice.repo import load_repository
from dice.ast import make_parser
from dice.info import new_info

import ujson
import typer

query_app = typer.Typer(help="Query the database")

@query_app.command()
def query(
    hosts: str = typer.Argument(
        "0.0.0.0/0",
        help="list of hosts or ranges to query for"
    ),
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
    fields: Annotated[str, typer.Option()] = "hosts,ports,services,labels,tags", 
    exclude: Annotated[str, typer.Option()] = "",
    delimeter: Annotated[str, typer.Option()] = ","
) -> None:
    hlist = hosts.split(delimeter)
    parser = make_parser()
    qt = parser.to_sql(hlist, q)

    repo = load_repository(db=database)
    n, batches = repo.query(qt)
    print(f"found {n} hosts")

    flist = fields.split(",")
    if exclude:
        flist = list(set(flist) - set(exclude.split(",")))

    con = repo.connect()
    info_b = new_info(flist)
    for b in batches:
        ips = b.ip.tolist()
        iq = info_b.make(ips)
        df = con.execute(iq).df()
        df['services'] = df['services'].apply(
            lambda services: [
                {**s, 'data': ujson.loads(s['data'])} for s in services
            ]
        )
        print(df.to_json(orient="records", lines=True, force_ascii=False))

