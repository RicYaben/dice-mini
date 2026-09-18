from cyclopts import App

from .health import health as cmd

health = App(name="health")
health.default(cmd)
