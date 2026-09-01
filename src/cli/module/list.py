from dice.internal.recipe import new_builder


def list(
    modules: str = "*",
    registries: str | None = None,
) -> None:
    recipe = new_builder()
    if registries:
        recipe.registries(registries.split(","))

    recipe._cmanager.info(modules=modules.split(","))
