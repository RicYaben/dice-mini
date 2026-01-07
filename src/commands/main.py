import typer

from commands.query import query_app
from commands.run import run_app
from commands.add import insert_app
from commands.modules import modules_app
from commands.copy import copy_app
from commands.diff import diff_app
from commands.health import health_app

app = typer.Typer(help="dice-mini CLI")
app.add_typer(query_app)
app.add_typer(run_app)
app.add_typer(insert_app)
app.add_typer(modules_app, name="modules")
app.add_typer(copy_app)
app.add_typer(diff_app)
app.add_typer(health_app)

def main():
    app()