import cyclopts

app = cyclopts.App(name="dice-mini", help="dice-mini CLI")

app.command("cli.recipe.app:app")
app.command("cli.modules.app:app")
app.command("cli.search.app:app")
app.command("cli.source.app:app")

app.command("cli.compare:compare", name="compare")
app.command("cli.health:health", name="health")


def main():
    app()
