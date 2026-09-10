from dice.internal.recipe import new_builder

from .context import Context


def list(
    ctx: Context | None = None,
) -> None:
    if ctx is None:
        ctx = Context()

    recipe = new_builder().registries(ctx.registries)
    recipe._cmanager.info(modules=ctx.modules)
