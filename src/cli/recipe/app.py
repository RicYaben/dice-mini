from cyclopts import App
from cyclopts.config import Toml

from cli.recipe.recipe import recipe

app = App(
    config=Toml("config.toml"),
)

app.default(recipe)
app.command("cli.recipe.bake:bake")
app.command("cli.recipe.list:list")
app.command("cli.recipe.show:show")
#app.command("cli.recipe.remove:remove")
#app.command("cli.recipe.update:update")
