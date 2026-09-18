from typing import Annotated

from cyclopts import Parameter

from dice.recipes import cookbook, prepare
from dice.results import results

from .context import RecipeContextOptions, configure_context


def recipe(
    fpath: Annotated[
        str,
        Parameter(
            name=["--fpath", "-f"],
            help="Path to the recipe file. Can be a local file, a URL, or recipe ID. Note: Recipe IDs must be registered in the cookbook",
        ),
    ],
    opts: RecipeContextOptions | None = None,
) -> None:
    conf = configure_context(opts)
    cb = cookbook(conf.databases.cookbook)
    rp = cb.resolve(fpath)

    wf = (
        prepare(rp)
        .registries(conf.modules.registries)
        .flags(**conf.modules.flags.root)
        .bake()
    )

    repo = results(conf.databases.results)
    wf.start(repo)
