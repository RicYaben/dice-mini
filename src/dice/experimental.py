from dataclasses import dataclass, field
from typing import Any, Literal
from sqlalchemy import select, func, and_
from sqlalchemy.dialects import sqlite

from dice.shared.models import Model

Op = Literal["eq", "ne", "gt", "gte", "lt", "lte", "in", "bt"]

@dataclass(frozen=True)
class Condition:
    field: str
    op: Op
    value: Any
    json: bool = False

def compile_condition(model, c: Condition):
    # JSON detection
    if "." in c.field:
        root, *path = c.field.split(".")
        json_path = "$." + ".".join(path)

        col = getattr(model, root)
        expr = func.json_extract(col, json_path)
    else:
        expr = getattr(model, c.field)

    # operators
    if c.op == "eq":
        return expr == c.value
    if c.op == "ne":
        return expr != c.value
    if c.op == "gt":
        return expr > c.value
    if c.op == "gte":
        return expr >= c.value
    if c.op == "lt":
        return expr < c.value
    if c.op == "lte":
        return expr <= c.value
    if c.op == "in":
        return expr.in_(c.value)
    if c.op == "bt":
        a, b = c.value
        return expr.between(a, b)

    raise ValueError(f"Unknown op {c.op}")

def parse_condition(key: str, value: Any) -> Condition:
    # operator split
    if "__" in key:
        field, op = key.split("__", 1)
    else:
        field, op = key, "eq"
    return Condition(field=field, op=op, value=value)

@dataclass
class Query:
    model: Any
    conditions: list[Condition] = field(default_factory=list)

    def where(self, **kwargs) -> "Query":
        for k, v in kwargs.items():
            self.conditions.append(parse_condition(k, v))
        return self

def build_query(q: Query):
    model = q.model

    stmt = select(model)

    if q.conditions:
        stmt = stmt.where(
            and_(
                *[
                    compile_condition(model, c)
                    for c in q.conditions
                ]
            )
        )

    return stmt

def to_sql(stmt) -> str:
    return str(
        stmt.compile(
            dialect=sqlite.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )

def query(model: type[Model], **clauses) -> str:
    q = Query(model).where(**clauses)
    return to_sql(build_query(q))

