import typer

from dice.internal.components import new_component_manager
from dice.internal.config import load_configuration
from dice.internal.modules import load_registry_plugins
from modules import registry

modules_app = typer.Typer(help="Check registered modules")


def parse_params(params: str, delimiter: str) -> dict[str, str]:
    r: dict[str, str] = {}
    p_split = params.split(delimiter)
    for p in p_split:
        k, val = p.split("=", 1)
        r[k] = val
    return r


@modules_app.command()
def config(
    m: str = typer.Option(
        "*", "-M", "--modules", help="Comma separated list of modules"
    ),
    params: str = typer.Option(
        "",
        "-p",
        "--params",
        help="Parameters to set. Example: 'key=value;key=value;key=value'",
    ),
    delimiter: str = typer.Option(
        ";", "-d", "--delimiter", help="Delimiter for the parameters"
    ),
    plugins: str | None = typer.Option(
        None, "--plugins", help="Load module registries as plugins"
    ),
    configuration: str = typer.Option(
        "modules.toml", "-c", "--configuration", help="Path to configuration file"
    ),
) -> None:
    if not params:
        print("nothing to configure: parameters required")
        return

    manager = new_component_manager("-")
    manager.register(registry)

    # registry plugins
    if plugins and (regs := load_registry_plugins(plugins)):
        for r in regs:
            manager.register(r)

    modules = manager.get_modules(modules=m.split(","))
    conf = load_configuration(configuration)

    # otherwise, for each parameter, attempt to set the value
    kwargs = parse_params(params, delimiter)
    for mod in modules:
        conf.update(mod.desc.name, **kwargs)
    conf.dump(configuration)
