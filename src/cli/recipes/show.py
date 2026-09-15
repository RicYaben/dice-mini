from dice.recipes import cookbook

from .context import RecipesContext


def show(
    ctx: RecipesContext,
):
    cb = cookbook(ctx.cookbook)
    for recipe in cb.search(ctx.recipes):
        r = cb.resolve(recipe.path)
        print(r.to_dict())
