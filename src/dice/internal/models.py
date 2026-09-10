from sqlalchemy import UniqueConstraint

from dice.shared.models import Model


class CookbookModel(Model):
    pass


class RecipeRef(CookbookModel, table=True):
    name: str
    path: str

    __table_args__ = (UniqueConstraint("name", "path"),)
