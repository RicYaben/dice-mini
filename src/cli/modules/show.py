from dice.internal.recipe import new_builder

from .context import Context


def show(
    ctx: Context | None = None,
) -> None:
    if ctx is None:
        ctx = Context()
    recipe = new_builder().registries(ctx.registries)
    for mod in recipe._cmanager.get_modules(modules=ctx.modules):
        print(str(mod.desc), "\n")
