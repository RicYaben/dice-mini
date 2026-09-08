from dataclasses import dataclass
from typing import Annotated

from config.params import (
    ConfigOptions,
    DatabaseOptions,
    ModuleOptions,
    Parameter,
)
from dice.internal.config import DiceConfig


@dataclass
class RecipeContext:
    config: Annotated[ConfigOptions, Parameter(group="Configuration")]
    modules: Annotated[ModuleOptions, Parameter(group="Modules")]
    database: Annotated[DatabaseOptions, Parameter(group="Databases")]


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


def configure_context(ctx: RecipeContext) -> DiceConfig:
    conf = (
        DiceConfig.load(ctx.config.configuration)
        if ctx.config.configuration
        else DiceConfig()
    )

    conf = conf.override(
        **ctx.config.overrides(),
        modules=ctx.modules.overrides(),
        databases=ctx.database.overrides(),
    )

    return conf
