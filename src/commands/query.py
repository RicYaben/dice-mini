import typer
from typing_extensions import Annotated

from dice.repo import load_repository
from dice.ast import make_parser
from dice.info import new_info

query_app = typer.Typer(help="Query the database")

@query_app.command()
def query(
    q: str, 
    database: str, 
    fields: Annotated[str, typer.Option()] = "hosts,ports,services,labels,tags", 
    exclude: Annotated[str, typer.Option()] = ""
) -> None:
    parser = make_parser()
    q = parser.to_sql(q)

    repo = load_repository(db=database)
    n, batches = repo.queryb(q)
    print(f"found {n} hosts")

    flist = fields.split(",")
    if exclude:
        flist = list(set(flist) - set(exclude.split(",")))

    con = repo.get_connection()
    info_b = new_info(flist)
    for b in batches:
        ips = b.ip.tolist()
        iq = info_b.make(ips)
        df = con.execute(iq).df()
        print(df.to_json(orient="records", lines=True, force_ascii=False))

