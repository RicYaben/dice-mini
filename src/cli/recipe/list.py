from dice.cli.tools import load_cookbook


def list(
    recipes: list[str] | None = None,
    cookbook: str | None = None,
):
    cb = load_cookbook(cookbook)
    for recipe in cb.search(recipes):
        print(recipe)
