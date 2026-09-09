from dice.internal.recipe import new_builder


def show(
    modules: str = "*",
    registries: str | None = None,
) -> None:
    recipe = new_builder()
    if registries:
        recipe.registries(registries.split(","))

    for mod in recipe._cmanager.get_modules(modules=modules.split(",")):
        print(str(mod.desc), "\n")
