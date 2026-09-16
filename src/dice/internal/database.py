from collections.abc import Iterable, Sequence
from pathlib import Path
from sqlite3 import IntegrityError
from typing import Any, Literal

from sqlalchemy import Connection, Engine, Row
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlmodel import Session, SQLModel, create_engine, insert, select

from dice.shared.models import DatabaseModel


def insert_records(
    session: Session, model: type[DatabaseModel], records: list[dict]
) -> Sequence[Row]:
    """
    Quickest way to blindly insert thousands of records into a database.
    Mainly used by the sourcerer to insert actual records.
    https://github.com/fastapi/sqlmodel/discussions/659
    """
    if not records:
        return []

    result = session.exec(insert(model), params=records).all()
    if result is None:
        return []

    session.commit()
    return result


def insert_or_ignore(
    session: Session,
    model: type[DatabaseModel],
    items: Iterable[DatabaseModel],
) -> Sequence[Row]:
    items = list(items)
    if not items:
        return []

    rows_data = [item.model_dump(exclude_unset=True) for item in items]

    stmt = (
        sqlite_insert(model).values(rows_data).prefix_with("OR IGNORE").returning(model)
    )

    result = session.exec(stmt).all()
    if result is None:
        return []

    session.commit()
    session.exec(select(model).execution_options(populate_existing=True))
    return result


def get_or_create(session: Session, model: type[DatabaseModel], **kwargs) -> tuple[Any, bool]:
    # Try to get existing
    obj = session.exec(select(model).filter_by(**kwargs)).first()  # type: ignore
    if obj:
        return obj, False

    # Try to insert
    obj = model(**kwargs)
    session.add(obj)
    try:
        session.commit()
        session.refresh(obj)
        return obj, True
    except IntegrityError:
        session.rollback()
        # Another process created it first
        obj = session.exec(select(model).filter_by(**kwargs)).one()
        return obj, False


class Connector:
    def __init__(
        self,
        location: str | Path | None,
        model: type[SQLModel] | None = None,
        driver: Literal["sqlite"] = "sqlite",
    ) -> None:
        self.location: str | Path = location if location else ":memory:"
        self.driver: str = driver
        self.engine: Engine | None = None
        self.model: type[SQLModel] | None = model

    def load(self):
        e = create_engine(f"{self.driver}:///{self.location}")
        if self.model:
            self.model.metadata.create_all(e)

        self.engine = e
        return e

    def connection(self) -> Connection:
        if not self.engine:
            _ = self.load()

        assert self.engine is not None
        return self.engine.connect()  # type: ignore

    def session(self) -> Session:
        return Session(self.connection())


def new_connector(db: str | Path |None, model: type[SQLModel] | None = None, name: Literal["sqlite"] = "sqlite") -> Connector:
    return Connector(db, driver=name, model=model)
