from cyclopts import App
from cyclopts.config import Toml

recipes = App(
    config=Toml("config.toml"),
)

recipes.command("cli.recipe.list:list")
recipes.command("cli.recipe.show:show")
# app.command("cli.recipe.remove:remove")
# app.command("cli.recipe.update:update")
