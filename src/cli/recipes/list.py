from dice.cli.tools import load_cookbook

from .context import RecipesContext


def list(
    ctx: RecipesContext,
):
    cb = load_cookbook(ctx.cookbook)
    for recipe in cb.search(ctx.recipes):
        print(recipe)
