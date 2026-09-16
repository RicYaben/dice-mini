from cyclopts import App

search = App(name="search", help="Query the database")
search.command("cli.search.search:search")
