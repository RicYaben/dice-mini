from pathlib import Path
from typing import Annotated

from cyclopts import Parameter

from dice.cli.config.args import ModulesArg
from dice.cli.config.models import CommandOptions
from dice.cli.tools import load_repository, writer
from dice.internal.recipe import new_builder

from .context import RecipeContextOptions, configure_context


def bake(
    cmds: CommandOptions,
    opts: RecipeContextOptions | None = None,
    modules: ModulesArg | None = None,
    run: Annotated[
        bool,
        Parameter(
            name=["--run", "-r"],
            help="Run the recipe after baking",
        ),
    ] = False,
    store: Annotated[
        Path | None,
        Parameter(
            name=["--store", "-s"],
            help="Path to store the baked workflow",
        ),
    ] = None,
):
    conf = configure_context(opts)
    wf = (
        new_builder()
        .registries(conf.modules.registries)
        .components(cmds.resolve(), modules)
        .flags(**conf.modules.flags.root)
        .bake()
    )

    with writer(store) as w:
        w.write(wf.dump() + "\n")

    if run:
        repo = load_repository(db=conf.databases.results)
        wf.start(repo)
