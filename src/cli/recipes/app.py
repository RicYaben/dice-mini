from cyclopts import App
from cyclopts.config import Toml

recipes = App(
    name="recipes",
    config=Toml("config.toml"),
)

recipes.command("cli.recipes.list:list")
recipes.command("cli.recipes.show:show")
