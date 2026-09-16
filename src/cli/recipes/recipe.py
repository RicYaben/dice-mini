from typing import Annotated

from cyclopts import Parameter

from dice.cli.tools import load_cookbook, load_repository
from dice.internal.recipe import prepare

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
    cb = load_cookbook(conf.databases.cookbook)
    rp = cb.resolve(fpath)

    wf = (
        prepare(rp)
        .registries(conf.modules.registries)
        .flags(**conf.modules.flags.root)
        .bake()
    )

    repo = load_repository(conf.databases.results)
    wf.start(repo)
