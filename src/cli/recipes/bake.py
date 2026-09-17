import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Annotated

from cyclopts import Parameter

from dice.cli.args import ModulesArg
from dice.cli.models import CommandOptions
from dice.recipes import workflow
from dice.results import results

from .context import RecipeContextOptions, configure_context


def writer_ctx(output: str | Path | None) -> object:
    if not output:
        return nullcontext(sys.stdout)
    return open(output, "+a")


class Writer:
    def __init__(self, output: str | Path | None):
        self._ctx = writer_ctx(output)

    def __enter__(self):
        return self._ctx.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._ctx.__exit__(exc_type, exc_val, exc_tb)


def writer(output: str | Path | None) -> Writer:
    return Writer(output)


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
        workflow()
        .registries(conf.modules.registries)
        .components(cmds.resolve(), modules)
        .flags(**conf.modules.flags.root)
        .bake()
    )

    with writer(store) as w:
        w.write(wf.dump() + "\n")

    if run:
        res = results(conf.databases.results)
        wf.start(res)
