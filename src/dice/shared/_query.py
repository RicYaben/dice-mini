from typing import Any, Optional
from sqlalchemy.dialects import sqlite

from dice.shared.models import Model

def parse_clause(clause: str, value: Any) -> str:
    # Operators
    ops = {
        "gt": ">",
        "lt": "<",
        "gte": ">=",
        "lte": "<=",
        "ne": "!=",
        "eq": "=",
        "in": "IN",
        "bt": "BETWEEN",
    }

    # Extract field and operator
    if "__" in clause:
        field, op = clause.split("__", 1)
        modifier = ops.get(op, "=")
    else:
        field, modifier = clause, "="

    # BETWEEN
    if modifier == "BETWEEN":
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("BETWEEN operator requires a 2-element list/tuple")
        low, high = value
        return f'"{field}" BETWEEN {low} AND {high}'

    # IN
    if isinstance(value, list):
        vals = ", ".join(f"'{v}'" if isinstance(v, str) else str(v) for v in value)
        return f'"{field}" IN ({vals})'

    # String
    if isinstance(value, str):
        return f"\"{field}\" {modifier} '{value}'"

    # Numeric
    return f'"{field}" {modifier} {value}'

def parse_json_clause(field: str, op: str, value: Any) -> str:
    json_path = f"$.{field}"

    # NULL handling
    if value is None:
        if op == "ne":
            return f"json_extract(data, '{json_path}') IS NOT NULL"
        if op == "eq":
            return f"json_extract(data, '{json_path}') IS NULL"

    # operators
    ops = {
        "gt": ">",
        "lt": "<",
        "gte": ">=",
        "lte": "<=",
        "ne": "!=",
        "eq": "=",
    }

    sql_op = ops.get(op, "=")

    if isinstance(value, str):
        value = f"'{value}'"

    return f"json_extract(data, '{json_path}') {sql_op} {value}"

def with_clauses(q: str, clauses: Optional[dict] = None) -> str:
    qc = ""
    if clauses:
        qc = "WHERE " + " AND ".join(parse_clause(k, v) for k, v in clauses.items())
    return q.format(clauses=qc)

def query(table: Model | str, fields: list[str]=["*"], **clauses) -> str:
    if isinstance(table, Model):
        table = table.__tablename__
        
    return with_clauses(
        f"""
        SELECT {",".join(fields)}
        FROM {table}
        {{clauses}}
        """,
        clauses,
    )

def to_sql(stmt) -> str:
    return str(
        stmt.compile(
            dialect=sqlite.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )