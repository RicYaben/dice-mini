from dice.cli.tools import load_cookbook

from .context import RecipesContext


def show(
    ctx: RecipesContext,
):
    cb = load_cookbook(ctx.cookbook)
    for recipe in cb.search(ctx.recipes):
        r = cb.resolve(recipe.path)
        print(r.to_dict())
