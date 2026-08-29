import typer
import ujson

from dice.cli.tools import load_cookbook, load_repository, writer
from dice.internal.recipe import new_builder, prepare
from dice.shared.modules import MFACTORY, ModuleType


def parse_command(cmd: str):
    ts = MFACTORY.all()
    mc = MFACTORY.get(cmd)
    return ts[ts.index(mc) :]

def parse_component_modules(command: str | None, components: str | None, modules: str | None) -> tuple[list[ModuleType], list[str]]:
    comps = [c.strip() for c in components.split(",") if c.strip()] if components else []
    mods = [m.strip() for m in modules.split(",") if m.strip()] if modules else []

    c = parse_command(command) if command else [MFACTORY.get(c) for c in comps]
    if not (command or components):
        c = MFACTORY.get("s")

    return c, mods

recipe_app = typer.Typer(help="Recipes")

@recipe_app.callback(
    help="Run a recipe, either local or remote.",
    invoke_without_command=True
)
def recipe(
    ctx: typer.Context,
    fpath: str | None = typer.Option(
        None, "-R", "--recipe",
        help="Path to recipe, identifier, or remote URL",
    ),
    cookbook: str | None = typer.Option(
        None, "-cb", "--cookbook", help="Path to cookbook"
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
    params: str = typer.Option(
        "{}", "--params", help="Parameters to set, as JSON"
    ),
) -> None:
    if ctx.invoked_subcommand is not None:
        return

    if fpath is None:
        raise typer.BadParameter(
            "Missing recipe path, identifier, or remote URL",
            param_hint="fpath",
        )

    cb = load_cookbook(cookbook)
    desc = cb.resolve(fpath)
    plugs = plugins.split(",") if plugins else None

    recipe = (
        prepare(desc)
        .plugins(plugs)
        .configure(configuration)
        .params(**ujson.loads(params))
        .bake()
    )

    repo = load_repository(database)
    recipe.start(repo)


@recipe_app.command(help="Bake a recipe from arguments")
def bake(
    # One of S,F,C,T; in that ascending order, it runs all components until that one (inclusive)
    command: str | None = typer.Argument(None, help="components to use"),
    # Components and modules to use
    components: str | None = typer.Option(
        None, "-C", "--components", help="list of components to load"
    ),
    modules: str = typer.Option("*", "-M", "--modules", help="modules to load"),
    plugins: str | None = typer.Option(
        None, "-p", "--plugins", help="Load module registries as plugins"
    ),
    # Configuration options for modules
    configuration: str | None = typer.Option(
        None, "-c", "--configuration", help="Load a configuration"
    ),
    params: str = typer.Option(
        "{}", "--params", help="Parameters to set, as JSON"
    ),
    store: str = typer.Option("", "-s", "--store", help="where to store the recipe. stdout by default"),
    # Whether to run the recipe. Sub-command for running the recipe.
    run: bool = typer.Option(
        False, "--run", help="Run the recipe"
    ),
    database: str | None = typer.Option(
        None, "-db", "--database", help="path to database"
    ),
):

    comps, mods = parse_component_modules(command, components, modules)
    plugs = plugins.split(",") if plugins else None
    recipe = (
        new_builder()
        .plugins(plugs)
        .components(comps, mods)
        .configure(configuration)
        .params(**ujson.loads(params))
        .bake()
    )

    with writer(store) as w:
        w.write(recipe.dump() + "\n")

    if run:
        repo = load_repository(db=database)
        recipe.start(repo)


@recipe_app.command(name="list",help="query a cookbook for recipes. Returns a list of matching recipes")
def list(
    r: str = typer.Option("*", "-R", "--recipes", help="Comma separated list of recipes"),
    loc: str | None = typer.Option(None, "-cb", "--cookbook", help="where recipes are stored"),
    output: str = typer.Option("", "-o", "--output", help="where to output results. stdout by default"),
):

    with writer(output) as w:
        cb = load_cookbook(loc)
        df = cb.search(r.split(",")).df()
        w.write(df.to_string(index=False) + "\n")

@recipe_app.command()
def show(
    r: str = typer.Option("*", "-R", "--recipes", help="Comma separated list of recipes"),
    loc: str | None= typer.Option(None, "-cb", "--cookbook", help="where recipes are stored"),
    output: str = typer.Option("", "-o", "--output", help="where to output results. stdout by default"),
):

    with writer(output) as w:
        cb = load_cookbook(loc)
        for ref in cb.find(r.split(",")):
            desc = cb.resolve(ref.path)
            w.write(desc.to_dict(), end="\n")

@recipe_app.command()
def update(
    r: str = typer.Option("*", "-R", "--recipes", help="Comma separated list of recipes"),
    loc: str | None = typer.Option(None, "-cb", "--cookbook", help="where recipes are stored"),
):
    ...

@recipe_app.command()
def remove(
    r: str = typer.Option("*", "-R", "--recipes", help="Comma separated list of recipes"),
    loc: str | None = typer.Option(None, "-cb", "--cookbook", help="where recipes are stored"),
):
    ...
