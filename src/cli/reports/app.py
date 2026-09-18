from cyclopts import App

reports = App(name="report", help="Generate a report from search results")
reports.command("cli.reports.compare:compare_cmd", name="compare")
