import pandas as pd

from dataclasses import dataclass

from dice.config import CLASSIFIER
from dice.module import Module, new_module
from dice.query import query_db
from dice.records import Access, Authentication, Connection, Maturity, Record, make_record

def vuln_cls_init(mod: Module) -> None:
    # Encryption
    mod.register_label(
        "unencrypted-communication",
        "unencrypted communication established",
        level=2,
    )
    # Authentication
    mod.register_label(
        "anonymous-authentication",
        "allows unauthenticated clients to connect",
        level=3,
    )
    mod.register_label(
        "self-signed-certificate-authentication",
        "allows anonymous clients to connect using self-signed certificates",
        level=3,
    )
    # Authorization
    mod.register_label(
        "read-access",
        "grants anonymous clients read access",
        level=4,
    )
    mod.register_label(
        "write-access",
        "grants anonymous clients write rights",
        level=5,
    )
    mod.register_label(
        "execute-access",
        "grants anonymous clients execute rights",
        level=5,
    )
    # Vulnerabilities
    mod.register_label(
        "neglected",
        "service running on major version behind the current reccommendation",
        level=4,
    )
    mod.register_label(
        "obsolete",
        "service running on deprecated or onbsolete firmware",
        level=5,
    )
    mod.register_label(
        "cves",
        "service contains a known CVE",
        level=5,
    )

@dataclass
class CPE:
    last_version: str
    name: str
    maturity: Maturity

def find_cpe(mod: Module, vendor: str = "*", product: str = "*") -> CPE | None:
    # 1. check the database first
    qcpe = "..."
    if row:= mod.repo().get_connection().execute(qcpe).fetchone():
        return CPE(last_version=row[0], name=row[1], maturity=row[2])
    
    # 2. check the NVDE
    #cpe = f"cpe:a:{vendor.lower()}:{product.lower()}:-:*:*:*:*:*:*:*"
    raise NotImplementedError

def fetch_cves(mod: Module, cpe: CPE) -> pd.DataFrame:
    # check the db for related cve's or fech from the NVDE or CVE api
    raise NotImplementedError


def cpe_cls_handler(mod: Module) -> None:
    # 1. group by name of vendor and product/service
    q = "..."
    # 2. fetch all of those from the database
    gen = mod.query(q)
    # 3. iterate, and for each,
    for ch in gen:
        assert isinstance(ch, pd.DataFrame)

        cpe = find_cpe(mod, ch["vendor"][0], ch["product"][0])
        if not cpe:
            continue

        # 3.1. eval deprecated, and set "obsolete" label if needed
        if cpe.maturity is Maturity.DEPRECATED:
            for fid in ch["id"].tolist():
                mod.store(mod.make_label(fid, "obsolete"))

        neglected = ch[ch["version"] < cpe.last_version]["id"].tolist()
        for fid in neglected:
            mod.store(mod.make_label(fid, "neglected"))

        cves: pd.DataFrame = fetch_cves(mod, cpe)
        if cves.empty:
            return
        
        for v, group in ch.groupby("version"):
            # rows where this version is affected
            mask = (cves["from"] <= v) & (cves["to"] >= v)
            vcves = cves.loc[mask]

            if vcves.empty:
                continue

            for fid in group["id"]:
                for _, row in vcves.iterrows():
                    mod.store(
                        mod.make_label(fid, "cve", row["cve"])
                    )
    

def vulnerable_cls_handler(mod: Module) -> None:
    def handler(row: pd.Series):
        fid = row["id"]
        record: Record = make_record(row)

        # connected
        if record.connection is not Connection.CONNECTED:
            return
        
        # encryption
        if not record.encryption:
            mod.store(mod.make_label(fid, "unencrypted-communication"))

        # authentication
        match record.authentication:
            case Authentication.UNAUTHORIZED:
                return
            case Authentication.ANONYMOUS:
                mod.store(mod.make_label(fid, "anonymous-authentication"))
            case Authentication.SELF_SIGNED_CERTIFICATE:
                mod.store(mod.make_label(fid, "self-signed-certificate-authentication"))

        # authorization
        acc = record.access
        if Access.WRITE in acc:
            mod.store(mod.make_label(fid, "write-access"))
        if Access.READ in acc:
            mod.store(mod.make_label(fid, "read-access"))
        if Access.EXECUTE in acc:
            mod.store(mod.make_label(fid, "execute-access"))

    q = query_db("fingerprints")
    mod.itemize(q, handler, orient="rows")

def make_classifier() -> Module:
    return new_module(CLASSIFIER, "vulnerable", vulnerable_cls_handler, vuln_cls_init)