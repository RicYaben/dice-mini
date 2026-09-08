import cyclopts

app = cyclopts.App(name="dice-mini", help="dice-mini CLI")

app.command("cli.recipe.recipe:recipe")
app.command("cli.recipe.bake:bake")

app.command("cli.recipe.app:recipes")
app.command("cli.modules.app:modules")
app.command("cli.source.app:sources")

app.command("cli.search.app:search")
app.command("cli.compare.app:compare")
app.command("cli.health.app:health")


def main():
    app()
