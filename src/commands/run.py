from typing import Optional
from dice.config import MFACTORY, DEFAULT_MODULES_DIR
from dice.module import load_registry_plugins, new_component_manager, load_registry
from dice.modules import registry as core

import dice
import typer

from dice.repo import load_repository

def parse_command(cmd: str):
    ts = MFACTORY.all()
    mc = MFACTORY.get(cmd)
    return ts[ts.index(mc):]

run_app = typer.Typer(help="DICE mini runner")

@run_app.command()
def run(
        command: str | None = typer.Argument(None, help="components to use"),
        components: str = typer.Option(
            None, 
            "-C",
            "--components",
            help="list of components to load"
        ),
        modules: str = typer.Option(
            "*", 
            "-M",
            "--modules",
            help="modules to load"
        ),
        database: str | None = typer.Option(
            None, 
            "-db",
            "--database",
            help="path to database"
        ),
        id: str = typer.Option(
            "-", 
            "--id",
            help="assign an ID to results"
        ),
        info: bool = typer.Option(
            False, 
            "--info",
            help="mock run"
        ),
        registry: str | None = typer.Option(
            DEFAULT_MODULES_DIR,
            "--registry",
            help="Path to the directory containing module resitry"
        ),
        plugins: str | None = typer.Option(
            None,
            "--plugins",
            help="Load module registries as plugins"
        ),
        save: Optional[str] = typer.Option(
            None,
            "-s"
            "--sources-output",
            help="Path to sources directory. Save sources locally outside the database"
        )
    ):
    if not (command or components):
        raise Exception("command or componets required. DICE needs to know what to do")
    
    mods = [mod.strip() for mod in modules.split(",") if mod.strip()] if modules else []
    comps = [c.strip() for c in components.split(",") if c.strip()] if components else []
    cc = parse_command(command) if command else [MFACTORY.get(c) for c in comps]

    manager = new_component_manager(id)
    manager.register(core)

    # plugin modules
    if registry and (reg := load_registry(registry)):
        manager.register(reg)

    # registry plugins
    if plugins and (regs := load_registry_plugins(plugins)):
        for r in regs:
            manager.register(r)

    cb = manager.build(types=cc, modules=mods)
    engine = dice.new_engine(*cb)
    if info:
        manager.info(mods)
        engine.info()
        return

    repo = load_repository(db=database, save=save)
    engine.run(repo)
    