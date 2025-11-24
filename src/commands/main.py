import typer

from commands.query import query_app
from commands.run import run_app
from commands.add import insert_app
from commands.modules import modules_app

app = typer.Typer(help="dice-mini CLI")
app.add_typer(query_app)
app.add_typer(run_app)
app.add_typer(insert_app)
app.add_typer(modules_app, name="modules")

def main():
    app()