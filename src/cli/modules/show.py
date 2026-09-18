from dice.recipes import workflow

from .context import Context


def show(
    ctx: Context | None = None,
) -> None:
    if ctx is None:
        ctx = Context()
    recipe = workflow().registries(ctx.registries)
    # TODO: I do not like this API, would be better to make just a new manager and add the registries
    for mod in recipe._cmanager.get_modules(modules=ctx.modules):
        print(str(mod.desc), "\n")
