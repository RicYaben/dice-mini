import pandas as pd
import duckdb

from dice.models import Source, Label, Fingerprint, FingerprintLabel
from dice.helpers import new_collection

class Repository:
    def __init__(self, con: duckdb.DuckDBPyConnection) -> None:
        self._con = con

    def _query(self, table: str, **kwargs) -> pd.DataFrame:
        q = f"SELECT * FROM {table}"
        clauses = []
        params = []

        for k, val in kwargs.items():
            if isinstance(val, list):
                placeholders = ", ".join(["?"] * len(val))
                clauses.append(f"{k} IN ({placeholders})")
                params.extend(val)
            else:
                clauses.append(f"{k} = ?")
                params.append(val)

        if clauses:
            q += " WHERE " + " AND ".join(clauses)

        return self._con.execute(q, params).df()
    
    def _rebuild_records_view(self) -> None:
        tables = self._con.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_name LIKE 'records_%'
            ORDER BY table_name
        """).fetchall()

        union_sql = " UNION ALL ".join([f'SELECT * FROM "{t[0]}"' for t in tables])
        self._con.execute(f"CREATE OR REPLACE VIEW records AS {union_sql}")

    def get_fingerprints(self, *host: str, **kwargs) -> pd.DataFrame:
        'query the repo for records'
        # select all hosts
        if len(host):
            kwargs["host"] = list(host)
        return self._query("fingerprints", **kwargs)
    
    def get_records(self, *host: str, **kwargs) -> pd.DataFrame:
        'query the repo for records'
        # select all hosts
        if len(host):
            kwargs["ip"] = list(host)
        
        return self._query("records", **kwargs)
    
    def summary(self) -> dict:
        'returns a brief summary of the database contents'
        fpd = self._con.execute("""
            SELECT COUNT(DISTINCT host)
            FROM fingerprints       
        """).fetchone()

        lbld = self._con.execute("""
            SELECT COUNT(DISTINCT fp.host)
            FROM fingerprints fp
            JOIN fingerprint_labels fl ON fl.fingerprint_id = fp.id
        """).fetchone()

        summary = {}

        if fpd is not None:
            summary["fingerprinted"] = fpd[0]

        if lbld is not None:
            summary["labelled"] = lbld[0]
        
        return summary
    
    def add_records(self, tab: str, records: pd.DataFrame) -> None:
        'add a new records table to the db and rebuild the view'
        records.to_sql(tab, self._con, index=False)
        self._rebuild_records_view()

    def add_hosts(self, hosts: pd.DataFrame) -> None:
        hosts.to_sql("hosts", self._con, if_exists="append", index=False)
    
    def add_labels(self, *label: Label) -> None:
        'create new labels in the database'
        c = new_collection(*label)
        c.to_df().to_sql("labels", self._con, if_exists="append", index=False)

    def fingerprint(self, *fingerprint: Fingerprint) -> None:
        'fingerprint hosts. The dataframe is a collection of fingerprints'
        c = new_collection(*fingerprint)
        c.to_df().to_sql("fingerprints", self._con, if_exists="append", index=False)

    def label(self, *fp_lab: FingerprintLabel) -> None:
        'label hosts. The dataframe is a dict of label_id and host_id'
        c = new_collection(*fp_lab)
        c.to_df().to_sql("fingerprint_labels", self._con, if_exists="append", index=False)

def load_repository(sources: list[Source], db: str|None=None) -> Repository:
    '''
    Create a repository from a list of sources.
    - db: name of the database - if not set, load in memory
    '''
    con = duckdb.connect(database=db if db is not None else ":memory:")
    repo = Repository(con)

    tables = {f"records_{src.name}_{src.id}": src.load() for src in sources}
    for tab, df in tables.items():
        repo.add_records(tab, df)

    return repo