from cyclopts import App

modules = App(name="modules")

modules.command("cli.modules.list:list")
modules.command("cli.modules.show:show")
