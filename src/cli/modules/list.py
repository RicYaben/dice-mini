from dice.recipes import workflow

from .context import Context


def list(
    ctx: Context | None = None,
) -> None:
    if ctx is None:
        ctx = Context()

    recipe = workflow().registries(ctx.registries)
    # TODO: I do not like this API, would be better to make just a new manager and add the registries
    recipe._cmanager.info(modules=ctx.modules)
