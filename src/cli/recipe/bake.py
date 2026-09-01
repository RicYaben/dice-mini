from pathlib import Path

from config.params import (
    ConfigOptionsArg,
    DatabaseOptionsArg,
    ModuleOptionsArg,
)
from dice.cli.tools import load_repository, writer
from dice.internal.config import DiceConfig
from dice.internal.recipe import new_builder
from dice.shared.modules import MFACTORY, ModuleType


def parse_command(cmd: str | None) -> list[ModuleType]:
    if not cmd:
        return []

    ts = MFACTORY.all()
    mc = MFACTORY.get(cmd)
    return ts[ts.index(mc) :]

def parse_component_modules(command: str | None, components: str | None, modules: str | None) -> tuple[list[ModuleType], list[str]]:
    comps = [c.strip() for c in components.split(",") if c.strip()] if components else []
    mods = [m.strip() for m in modules.split(",") if m.strip()] if modules else []

    c = parse_command(command) if command else [MFACTORY.get(c) for c in comps]
    if not (command or components):
        c = [MFACTORY.get("s")]

    return c, mods

def bake(
    command: str | None,
    components: str | None,
    modules: str | None,
    config: ConfigOptionsArg,
    mconf: ModuleOptionsArg,
    dconf: DatabaseOptionsArg,
    run: bool = False,
    store: Path | None = None,
):

    conf = (
        DiceConfig.load(config.configuration)
        if config.configuration
        else DiceConfig()
    )

    conf.override(
        **config.overrides(),
        modules=mconf.overrides(),
        databases=dconf.overrides(),
    )

    comps, mods = parse_component_modules(command, components, modules)
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
