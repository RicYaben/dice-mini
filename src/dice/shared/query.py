from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from sqlalchemy import and_, func, select
from sqlalchemy.dialects import sqlite

from dice.shared.models import Model

Op = Literal["eq", "ne", "gt", "gte", "lt", "lte", "in", "bt"]


@dataclass(frozen=True)
class Condition:
    field: str
    op: Op
    value: Any
    json: bool = False


_OPERATORS: dict[Op, Callable[[Any, Any], Any]] = {
    "eq": lambda expr, v: expr == v,
    "ne": lambda expr, v: expr != v,
    "gt": lambda expr, v: expr > v,
    "gte": lambda expr, v: expr >= v,
    "lt": lambda expr, v: expr < v,
    "lte": lambda expr, v: expr <= v,
    "in": lambda expr, v: expr.in_(v),
    "bt": lambda expr, v: expr.between(v[0], v[1]),
}


def resolve_field(model, field: str):
    if "." not in field:
        return getattr(model, field)

    root, *path = field.split(".")
    col = getattr(model, root)

    json_path = "$." + ".".join(path)
    return func.json_extract(col, json_path)


def compile_condition(model, c: Condition):
    try:
        expr = resolve_field(model, c.field)
        op_func = _OPERATORS[c.op]
        return op_func(expr, c.value)
    except KeyError:
        raise ValueError(f"Unknown operator: {c.op}")
    except AttributeError as e:
        raise ValueError(f"Invalid field: {c.field}") from e


def parse_condition(key: str, value: Any) -> Condition:
    field, op = key.split("__", 1) if "__" in key else (key, "eq")
    return Condition(field=field, op=op, value=value)  # type: ignore


@dataclass
class Query:
    model: Any
    conditions: list[Condition] = field(default_factory=list)
    fields: list[str] | None = None

    def where(self, **kwargs) -> "Query":
        for k, v in kwargs.items():
            self.conditions.append(parse_condition(k, v))
        return self

    def select(self, *fields: str) -> "Query":
        self.fields = list(fields)
        return self


def build_query(q: Query):
    model = q.model

    if q.fields:
        columns = [resolve_field(model, f) for f in q.fields]
        stmt = select(*columns)
    else:
        stmt = select(model)

    if q.conditions:
        stmt = stmt.where(and_(*[compile_condition(model, c) for c in q.conditions]))

    return stmt


def to_sql(stmt) -> str:
    return str(
        stmt.compile(
            dialect=sqlite.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )


def query(model: type[Model], fields: list[str] | None = None, **clauses: dict) -> str:
    q = Query(model, fields=fields).where(**clauses)
    return to_sql(build_query(q))
