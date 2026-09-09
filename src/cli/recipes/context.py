from collections.abc import Callable
from pathlib import Path
from typing import Annotated

from cyclopts import Token
from pydantic import BaseModel

from config.params import (
    ConfigOptions,
    DatabaseOptions,
    ModuleOptions,
    Parameter,
)
from dice.internal.config import DiceConfig


class RecipeContext(BaseModel):
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


def token_converter(delimiter: str) -> Callable:
    def handler(token: Token):
        return [x.strip() for x in token.value.split(delimiter) if x.strip()]

    return handler

class RecipesContext(BaseModel):
    recipes:  Annotated[
        list[str] | None,
        Parameter(
            name=["--recipes", "-r"], help="Comma-separated list of recipes",
            converter=token_converter(","),
        ),
    ] = None
    cookbook: Annotated[
        Path | None,
        Parameter(name=["--cookbook", "-cb"], help="Path to the cookbook file"),
    ] = None
