from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from dice.cli.config.args import token_converter
from dice.cli.config.models import (
    ConfigOptions,
    DatabaseOptions,
    ModuleOptions,
    Parameter,
)
from dice.internal.config import DiceConfig


@dataclass
class RecipeContextOptions:
    config: Annotated[ConfigOptions | None, Parameter(name="*", group="Config")] = None
    modules: Annotated[ModuleOptions | None, Parameter(name="*", group="Modules")] = (
        None
    )
    database: Annotated[
        DatabaseOptions | None, Parameter(name="*", group="Database")
    ] = None


@dataclass
class RecipeContext:
    config: ConfigOptions
    modules: ModuleOptions
    database: DatabaseOptions


def create_context(
    config: ConfigOptions | None = None,
    modules: ModuleOptions | None = None,
    database: DatabaseOptions | None = None,
) -> RecipeContext:
    return RecipeContext(
        config=config or ConfigOptions(),
        modules=modules or ModuleOptions(),
        database=database or DatabaseOptions(),
    )


def configure_context(opts: RecipeContextOptions | None) -> DiceConfig:
    if not opts:
        return DiceConfig()

    ctx = create_context(opts.config, opts.modules, opts.database)
    conf = (
        DiceConfig.load(ctx.config.configuration)
        if ctx.config.configuration is not None
        else DiceConfig()
    )

    conf = conf.override(
        **ctx.config.to_dict(),
        modules=ctx.modules.to_dict(),
        databases=ctx.database.to_dict(),
    )

    return conf


@dataclass
class RecipesContext:
    recipes: Annotated[
        list[str] | None,
        Parameter(
            name=["--recipes", "-r"],
            help="Comma-separated list of recipes",
            converter=token_converter(","),
        ),
    ] = None
    cookbook: Annotated[
        Path | None,
        Parameter(name=["--cookbook", "-cb"], help="Path to the cookbook file"),
    ] = None
