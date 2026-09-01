from config.params import (
    ConfigOptionsArg,
    DatabaseOptionsArg,
    ModuleOptionsArg,
)
from dice.cli.tools import load_cookbook, load_repository
from dice.internal.config import DiceConfig
from dice.internal.recipe import prepare


def recipe(
    fpath: str,
    config: ConfigOptionsArg,
    mconf: ModuleOptionsArg,
    dconf: DatabaseOptionsArg,
) -> None:

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

    cb = load_cookbook(conf.databases.cookbook)
    rp = cb.resolve(fpath)

    wf = (
        prepare(rp)
        .registries(conf.modules.registries)
        .flags(**conf.modules.flags.root)
        .bake()
    )

    repo = load_repository(conf.databases.results)
    wf.start(repo)
