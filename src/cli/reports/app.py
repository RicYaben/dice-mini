from cyclopts import App

from .report import report

reports = App(name="report", help="Generate a report from search results")
reports.default(report)
reports.command("cli.reports.compare:compare_cmd", name="compare")
