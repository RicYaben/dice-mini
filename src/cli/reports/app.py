from cyclopts import App

from .report import report

reports = App(name="report", help="Generate a report from search results")
reports.default(report)
