from dice.recipes import cookbook

from .context import RecipesContext


def list(
    ctx: RecipesContext,
):
    cb = cookbook(ctx.cookbook)
    for recipe in cb.search(ctx.recipes):
        print(recipe)
