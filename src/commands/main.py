import typer

from commands.query import query_app
from commands.run import run_app
from commands.insert import insert_app

app = typer.Typer(help="dice-mini CLI")
app.add_typer(query_app)
app.add_typer(run_app)
app.add_typer(insert_app)

def main():
    app()