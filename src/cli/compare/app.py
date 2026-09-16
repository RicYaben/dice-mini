from cyclopts import App

from .compare import diff

compare = App(name="compare")

compare.default(diff)
