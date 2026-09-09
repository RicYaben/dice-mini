from enum import Enum
from pathlib import Path
from typing import Annotated

from cyclopts import Parameter

from dice.cli.tools import load_repository, writer
from dice.internal.recipe import new_builder
from dice.shared.modules import (
    ModuleType,
    filter_module_types,
    max_module_types,
    module_type_aliases,
)

from .context import RecipeContext, configure_context


class Mode(int, Enum):
    strict = 1
    normal = 2


def parse_component_modules(
    commands: str | None,
    modules: str | None,
    mode: Mode | None,
) -> tuple[list[ModuleType], list[str]]:

    mods = [m.strip() for m in modules.split(",") if m.strip()] if modules else []
    tps = [c.strip() for c in commands.split(",") if c.strip()] if commands else []

    match mode:
        case Mode.strict | None:
            mt = filter_module_types(tps)
        case Mode.normal:
            mt = max_module_types(tps)
    return mt, mods


def bake(
    ctx: RecipeContext,
    *,
    commands: Annotated[
        str | None,
        Parameter(
            name=["--commands", "-C"],
            choices=module_type_aliases(),
            help="Command-separated list of commands to execute",
        ),
    ] = None,
    modules: Annotated[
        str | None,
        Parameter(name=["--modules", "-M"], help="Comma-separated list of modules"),
    ] = None,
    mode: Annotated[
        Mode | None,
        Parameter(
            name=["--mode", "-m"],
            help="Mode to run the recipe in",
        ),
    ] = Mode.normal,
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
    comps, mods = parse_component_modules(commands, modules, mode)
    conf = configure_context(ctx)
    wf = (
        new_builder()
        .registries(conf.modules.registries)
        .components(comps, mods)
        .flags(**conf.modules.flags.root)
        .bake()
    )

    with writer(store) as w:
        w.write(wf.dump() + "\n")

    if run:
        repo = load_repository(db=conf.databases.results)
        wf.start(repo)
