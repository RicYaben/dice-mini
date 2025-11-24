import typer
from dice.module import new_component_manager
from dice.module import load_registry
from dice.modules import registry as dr

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
        "./mods",
        "--registry",
        help="Path to the directory containing module resitry"
    ),
) -> None:
    manager = new_component_manager("-")
    manager.add(*dr.all())

    # plugin modules
    if registry and (reg := load_registry(registry)):
        rmods = reg.all()
        manager.add(*rmods)

    modules = m.split(",")
    manager.list_modules(modules=modules)