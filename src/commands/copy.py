import typer

from dice.connector import new_connector  

copy_app = typer.Typer(help="Copy tables from one db to another")

@copy_app.command()
def copy(
    f: str = typer.Argument(help="Copy from"),
    t: str = typer.Argument(help="Copy to"),
    tables: str = typer.Option(
        None,
        "-t",
        "--tables",
        help="Tables to copy. Comma separated",
    ),
):
    if not tables:
        raise ValueError("copy needs one or more tables")
    
    tabs = [t.strip() for t in tables.split(",") if t.strip()]
    tcon = new_connector(t)
    tcon.attach(f, "src")

    for tab in tabs:
        print(f"copy {tab}")
        tcon.copy(tab, "src")
