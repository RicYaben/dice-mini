import warnings

import cyclopts
from cyclopts import Group

app = cyclopts.App("dice-mini", "dice-mini CLI")

core = Group(name="Core")
tools = Group(name="Tools")
reporting = Group("Reporting")

app.command("cli.recipes.recipe:recipe", group=core, help="Run a recipe")
app.command("cli.recipes.bake:bake", group=core, help="Create a new recipe")
app.command(
    "cli.reports.report:report", group=core, help="Search results and generate a report"
)
app.command(
    "cli.recipes.app:recipes", group=tools, help="List or show available recipes"
)
app.command(
    "cli.modules.app:modules", group=tools, help="List or show available modules"
)
app.command("cli.sources.app:sources", group=tools, help="Import sources")
app.command("cli.health.app:health", group=tools, help="Sanitize results")
app.command("cli.reports.app:reports", group=reporting, help="Generate reports")


def main():
    warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
    app()
