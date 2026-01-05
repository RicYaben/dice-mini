import typer
from dice.config import DEFAULT_MODULES_DIR
from dice.module import load_registry_plugins, new_component_manager, load_registry
from dice.modules import registry as core

modules_app = typer.Typer(help="Check registered modules")

@modules_app.command()
def list(
    m: str = typer.Option(
        "*",
        "-M",
        "--modules",
        help="Comma separated list of modules"
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
) -> None:
    manager = new_component_manager("-")
    manager.register(core)

    # registry discovery
    if registry and (reg := load_registry(registry)):
        manager.register(reg)

    # registry plugins
    if plugins and (regs := load_registry_plugins(plugins)):
        for r in regs:
            manager.register(r)

    modules = m.split(",")
    manager.info(modules=modules)