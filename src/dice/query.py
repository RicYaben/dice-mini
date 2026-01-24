from typing import Any

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

def query_vuln() -> str:
    q = """
WITH host_labels AS (
    SELECT
        f.host,
        list_distinct(list(l.name)) AS labels
    FROM fingerprints f
    JOIN fingerprint_labels fl
        ON fl.fingerprint_id = f.id
    JOIN labels l
        ON l.id = fl.label_id
    GROUP BY f.host
)
SELECT
    h.ip,
    hl.labels,
    r.name,
    r.domain,
    r.type,
    r.asn,
    r.as_name,
    r.as_domain,
    r.as_type,
    r.country
FROM host_labels hl
JOIN hosts h
    ON h.ip = hl.host
LEFT JOIN records_ipinfo_company r
    ON r.start_ip NOT LIKE '%:%'
   AND r.end_ip   NOT LIKE '%:%'
   AND inet_aton(h.ip)
       BETWEEN inet_aton(r.start_ip)
           AND inet_aton(r.end_ip)
ORDER BY inet_aton(h.ip) DESC
"""
    return q.strip()