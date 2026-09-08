import cyclopts

search = cyclopts.App(help="Query the database")
search.command("cli.search.search:search")
