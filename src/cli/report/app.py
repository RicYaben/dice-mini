from cyclopts import App

from .search import search

report = App(name="report", help="Generate a report from search results")
report.default(search)
