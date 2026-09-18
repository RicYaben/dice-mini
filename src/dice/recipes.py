from pathlib import Path

from dice.internal.database import new_connector
from dice.internal.recipes.cookbook import Cookbook, new_cookbook
from dice.internal.recipes.models import CookbookModel
from dice.internal.recipes.recipe import (
    Recipe,
    WorkflowBuilder,
    from_recipe,
    new_builder,
)
from dice.internal.results.repository import new_repository


def workflow() -> WorkflowBuilder:
    return new_builder()


def prepare(rc: Recipe):
    return from_recipe(rc)


def cookbook(res: str | Path | None) -> Cookbook:
    con = new_connector(res, CookbookModel)
    repo = new_repository(con)
    cb = new_cookbook(repo)
    return cb
