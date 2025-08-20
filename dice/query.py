import json

from dice.repo import Repository

def protocol_fingerprint(repo: Repository, fingerprint: str, port: int | None = None, protocol: str = "", prefix: str = "data") -> str:
    """
    Returns a prepared query to create a view with single protocol fingerprints.
    Expands the JSON `data` column into separate columns with optional prefix.
    """
    port_q = f"AND port = '{port}'" if port else ""
    proto_q = f"AND protocol = '{protocol}'" if protocol else ""

    con = repo.get_connection()

    # get one row to discover JSON keys
    row = con.execute(f"""
        SELECT data
        FROM fingerprints
        WHERE module_name = '{fingerprint}'
        {port_q}
        {proto_q}
        LIMIT 1
    """).fetchone()

    if not row:
        raise ValueError(f"No fingerprint data found for {fingerprint}")

    keys = json.loads(row[0]).keys()

    # build dynamic select list with prefix
    extracts = [
        f"json_value(f.data, '$.{k}') AS {prefix}_{k}" for k in keys
    ]

    return f"""
        SELECT
            f.* EXCLUDE (data),
            {', '.join(extracts)}
        FROM fingerprints f
        WHERE module_name = '{fingerprint}'
        {port_q}
        {proto_q}
    """