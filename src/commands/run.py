import typer
import ujson

from dice.cli.tools import load_repository
from dice.internal.components import new_component_manager
from dice.internal.config import load_configuration
from dice.internal.engine import new_engine
from dice.internal.modules import load_registry_plugins
from dice.shared.modules import MFACTORY
from modules import registry


def parse_command(cmd: str):
    ts = MFACTORY.all()
    mc = MFACTORY.get(cmd)
    return ts[ts.index(mc) :]


run_app = typer.Typer(help="DICE mini runner")


@run_app.command()
def run(
    command: str | None = typer.Argument(None, help="components to use"),
    components: str | None = typer.Option(
        None, "-C", "--components", help="list of components to load"
    ),
    modules: str = typer.Option("*", "-M", "--modules", help="modules to load"),
    database: str | None = typer.Option(
        None, "-db", "--database", help="path to database"
    ),
    plugins: str | None = typer.Option(
        None, "-p", "--plugins", help="Load module registries as plugins"
    ),
    configuration: str | None = typer.Option(
        None, "-c", "--configuration", help="Load a configuration"
    ),
    params: str | None = typer.Option(
        None, "--params", help="Parameters to set, as JSON"
    ),
):
    if not (command or components):
        command = "s"

    mods = [mod.strip() for mod in modules.split(",") if mod.strip()] if modules else []
    comps = (
        [c.strip() for c in components.split(",") if c.strip()] if components else []
    )
    cc = parse_command(command) if command else [MFACTORY.get(c) for c in comps]

    manager = new_component_manager("-")
    manager.register(registry)

    # registry plugins
    if plugins and (regs := load_registry_plugins(plugins)):
        for r in regs:
            manager.register(r)

    # build engine
    cb = manager.build(types=cc, modules=mods)
    engine = new_engine(*cb)

    # load configuration
    cmods = [m for _, m in manager.find(mods) if m.t in cc]
    if configuration:
        conf = load_configuration(configuration)
        for m in cmods:
            if f := conf.data[m.desc.name]:
                m.desc.flags.update(**f)

    # override with kwargs
    if params:
        mappings: dict[str, dict] = ujson.loads(params)
        for name, kwargs in mappings.items():
            if mod := next(filter(lambda x: x.desc.name == name, cmods)):
                mod.desc.flags.update(**kwargs)

    repo = load_repository(db=database)
    # TODO: ideally, the run returns some kind of summary report
    # so we can print it here
    res = engine.run(repo)
    # print(res.summary())

@run_app.command()
def info(
    command: str | None = typer.Argument(None, help="components to use"),
    components: str = typer.Option(
        None, "-C", "--components", help="list of components to load"
    ),
    modules: str = typer.Option("*", "-M", "--modules", help="modules to load"),
    plugins: str | None = typer.Option(
        None, "-p", "--plugins", help="Load module registries as plugins"
    ),
):
    if not (command or components):
        command = "s"

    mods = [mod.strip() for mod in modules.split(",") if mod.strip()] if modules else []
    comps = (
        [c.strip() for c in components.split(",") if c.strip()] if components else []
    )
    cc = parse_command(command) if command else [MFACTORY.get(c) for c in comps]

    manager = new_component_manager("-")
    manager.register(registry)

    # registry plugins
    if plugins and (regs := load_registry_plugins(plugins)):
        for r in regs:
            manager.register(r)

    # build engine
    cb = manager.build(types=cc, modules=mods)
    engine = new_engine(*cb)

    manager.info(mods)
    engine.info()


@run_app.command()
def recipe(
    id: str = typer.Argument(help="Identifier"),
    # TODO: add a list of trusted remote locations and a flag to get out of those
    remote: str | None = typer.Option(
        None, "-r", "--remote", help="Fetch specs from a remote location"
    ),
    database: str | None = typer.Option(
        None, "-db", "--database", help="Path to database"
    ),
    plugins: str | None = typer.Option(
        None, "-p", "--plugins", help="Load module registries as plugins"
    ),
    configuration: str | None = typer.Option(
        None, "-c", "--configuration", help="Load a configuration"
    ),
    params: str | None = typer.Option(
        None, "--params", help="Parameters to set, as JSON"
    ),
) -> None:
    # TODO: we load the recipe from either local or a remote location.
    # a recipe is a specification of how a scan should be performed.
    recipe = load_recipe(id, remote)
    if not recipe:
        return

    if configuration:
        conf = load_configuration(configuration)
        recipe.update(conf)

    if params:
        mappings: dict[str, dict] = ujson.loads(params)
        recipe.update(mappings)

    repo = load_repository(db=database)
    recipe.start(repo)
