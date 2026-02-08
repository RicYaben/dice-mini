
from typing import Any

# import ujson
# from dice.repo import Repository


# def protocol_fingerprint(
#     repo: Repository,
#     fingerprint: str,
#     port: int | None = None,
#     protocol: str = "",
#     prefix: str = "data",
#     expand: bool = True,
# ) -> str:
#     """
#     Returns a prepared query to create a view with single protocol fingerprints.
#     Expands the JSON `data` column into separate columns with optional prefix.
#     """
#     port_q = f"AND port = '{port}'" if port else ""
#     proto_q = f"AND protocol = '{protocol}'" if protocol else ""

#     con = repo.get_connection()

#     # get one row to discover JSON keys
#     row = con.execute(f"""
#         SELECT data
#         FROM fingerprints
#         WHERE module_name = '{fingerprint}'
#         {port_q}
#         {proto_q}
#         LIMIT 1
#     """).fetchone()

#     if not row:
#         raise ValueError(f"No fingerprint data found for {fingerprint}")

#     keys = ujson.loads(row[0]).keys()

#     # build dynamic select list with prefix
#     extracts = [f"json_value(f.data, '$.{k}') AS {prefix}_{k}" for k in keys]

#     return f"""
#         SELECT
#             f.* EXCLUDE (data),
#             LIST (l.name) AS labels,
#             {", ".join(extracts)}
#         FROM fingerprints f
#         WHERE module_name = '{fingerprint}'
#         {port_q}
#         {proto_q}
#         LEFT JOIN fingerprint_labels fl ON s.id = f.fingerprint_id
#         LEFT JOIN labels l ON fl.label_id = l.id
#         GROUP BY f.id
#     """


# def query_fingerprinter(
#     repo: Repository, expand: bool = True, prefix: str = "data_", **clauses
# ) -> str:
#     """returns a query for fingerprints w/o expanded data. NOTE: expanding here is faster than normalizing json elsewhere"""

#     q = """
#     SELECT {fields} 
#     FROM fingerprints AS f 
#     LEFT JOIN fingerprint_labels fl ON fl.fingerprint_id = f.id
#     LEFT JOIN labels AS l ON fl.label_id = l.id
#     {clauses}
#     GROUP BY f.*;
#     """

#     fields = ["f.*", "LIST (l.name) AS labels"]

#     qc = ""
#     if clauses:
#         qc = "WHERE " + " AND ".join(f"f.{k} = '{v}'" for k, v in clauses.items())

#     if not expand:
#         return q.format(fields=", ".join(fields), clauses=qc)

#     dq = f"SELECT data FROM fingerprints f {qc} LIMIT 1"
#     row = repo.get_connection().execute(dq).fetchone()
#     if not row:
#         raise ValueError(f"No fingerpritns found")

#     # parse JSON keys from that row
#     try:
#         data_json = ujson.loads(row[0])
#     except Exception as e:
#         raise ValueError(f"Failed to parse JSON from 'data' field: {e}")

#     # exclude original 'data' from f.* to avoid conflict
#     fields[0] += " EXCLUDE (data)"

#     # add a json_extract for every top-level key
#     for k in data_json.keys():
#         fields.append(f"json_extract(f.data, '$.{k}') AS {prefix}{k}")

#     return q.format(fields=", ".join(fields), clauses=qc)


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

    # Numeric (let DuckDB infer type)
    return f'"{field}" {modifier} {value}'


def with_clauses(q: str, clauses: dict | None = None) -> str:
    qc = ""
    if clauses:
        qc = "WHERE " + " AND ".join(parse_clause(k, v) for k, v in clauses.items())
    return q.format(clauses=qc)


def query_db(db: str, **clauses) -> str:
    return with_clauses(
        f"""
        SELECT *
        FROM {db}
        {{clauses}}
        """,
        clauses,
    )


def query_records(source: str, **clauses) -> str:
    return query_db(f"records_{source}", **clauses)


def query_serv_ports(zpcount_limit: int | None = None) -> str:
    limit_clause = ""
    if zpcount_limit:
        limit_clause = f"HAVING COUNT(DISTINCT z.sport) > {zpcount_limit}"
    return """
    SELECT
        f.host,
        COUNT(DISTINCT f.port) AS fpcount,
        COUNT(DISTINCT z.sport) AS zpcount,
        LIST(DISTINCT f.port) as fports,
        LIST(DISTINCT z.sport) as zports,
    FROM fingerprints AS f
    LEFT JOIN records_zmap AS z
        ON f.host = z.saddr
        -- remove tnis, our study contains traces of HART-IP, OPC UA, and S7
        AND z.sport NOT IN (102, 4840, 5094)
    GROUP BY f.host
    {clause}
    ORDER BY zports DESC
    """.format(clause=limit_clause)


def query_prefix_hosts(**clauses) -> str:
    return with_clauses(
        """
    SELECT
        h.prefix,
        COUNT(DISTINCT h.ip) AS count
    FROM hosts AS h
    {clauses}
    GROUP BY h.prefix
    """,
        clauses,
    )