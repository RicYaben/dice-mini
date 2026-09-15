import cyclopts
from cyclopts import Group

app = cyclopts.App("dice-mini", "dice-mini CLI")

core = Group(name="Core")
tools = Group(name="Tools")
reporting = Group("Reporting")

app.command("cli.recipes.recipe:recipe", group=core, help="Run a recipe")
app.command("cli.recipes.bake:bake", group=core, help="Create a new recipe")
app.command(
    "cli.recipes.app:recipes", group=tools, help="List or show available recipes"
)
app.command(
    "cli.modules.app:modules", group=tools, help="List or show available modules"
)
app.command("cli.sources.app:sources", group="D2", help="Import sources")
app.command("cli.health.app:health", group="D3", help="Sanitize results")
app.command("cli.report.app:reports", group=reporting, help="Generate reports")
# TODO: would be interesting to have reports.compare instead of this comparator between databases
# that way we can remove the need for a query? both functionalities can live together
app.command("cli.compare.app:compare", group=reporting, help="Compare results")


def main():
    app()
