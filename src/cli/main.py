import cyclopts

app = cyclopts.App("dice-mini", "dice-mini CLI")

app.command("cli.recipes.recipe:recipe", group="D1", help="Run a recipe")
app.command("cli.recipes.bake:bake", group="D1", help="Create a new recipe")

app.command("cli.recipes.app:recipes", group="D2", help="List or show available recipes")
app.command("cli.modules.app:modules", group="D2", help="List or show available modules")
app.command("cli.sources.app:sources", group="D2", help="Import sources")

app.command("cli.health.app:health", group="D3", help="Sanitize results")

app.command("cli.search.app:search", group="D4", help="Search results")
app.command("cli.compare.app:compare", group="D4", help="Compare results")

def main():
    app()
