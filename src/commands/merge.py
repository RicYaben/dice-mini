import typer

merge_app = typer.Typer(help="Merge multiple datasets")

@merge_app.command()
def merge(
    dbs: str = typer.Argument(help="Datasets to merge (comma-seaparated)"),
    sep: str = typer.Option(
        ",",
        "--separator",
        help="Separator"
    ),
    q: str = typer.Option(
        None,
        "-q",
        "--query",
        help="Merge subsets instead of the whole datasets"
    ),
    how: str = typer.Option(
        "default",
        "-h",
        "--how",
        help="Strategy to merge datasets. Options:default"
    ),
):
    raise NotImplementedError