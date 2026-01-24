from difflib import ndiff
from tqdm import tqdm

from dice.ast import make_parser
from dice.helpers import new_source
from dice.info import new_info
from dice.models import Source
from dice.repo import Repository

import pandas as pd
import numpy as np
import ujson

def diff_ports(left, right):
    if not isinstance(left, (np.ndarray)): 
        left = []
    if not isinstance(right, np.ndarray): 
        right = []

    if is_equal(left, right):
        return {}

    return {
        "added": [int(p) for p in right if p not in left],
        "removed": [int(p) for p in left if p not in right],
    }

def fmt_dif(old, new):
    return '\n'.join(ndiff(str(old).splitlines(), str(new).splitlines()))


def is_empty(v):
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    if isinstance(v, str) and v.strip() == "":
        return True
    if isinstance(v, (list, dict, tuple, set)) and len(v) == 0:
        return True
    return False

    
def is_equal(v1, v2):
    if isinstance(v1, (list, np.ndarray)) or isinstance(v2, (list, np.ndarray)):
        return np.array_equal(v1, v2)
    
    # Both empty? treat as equal
    if is_empty(v1) and is_empty(v2):
        return True
    
    # One empty, one not -> not equal
    if is_empty(v1) != is_empty(v2):
        return False
    
    # Normal equality, but fix NaN
    try:
        if pd.isna(v1) and pd.isna(v2):
            return True
    except:
        pass

    return v1 == v2

def changes(src: dict, dst: dict, fields: list[str] | None = None) -> dict[str, str]:
    'returns differences between fingerprints'
    if not fields:
        fields = list(src.keys())

    c = {}
    for k in fields:
        if not is_equal(src[k], dst[k]):
            c[k] = fmt_dif(src[k], dst[k])
    return c

def compare_services(name: str, left, right):
    match name:
        case "ethernetip":
            return changes(left, right, ["identities"])
        case "modbus":
            return changes(left, right)
        case "iec104":
            def ca100(x):
                return x["TypeID"] == 100

            la=list(filter(ca100, left["asdus"]))
            ra= list(filter(ca100,right["asdus"]))
            if not is_equal(la, ra):
                return {"asdus": fmt_dif(la, ra)}
            return {}

        case "fox":
            fields = left.keys() - ["timestamp", "result", "error", "status", "is_fox", "probe_status", "protocol"]
            return changes(left, right, fields)
        case _:
            return changes(left, right)
                        

def diff_services(left, right):
    """
    Compare two lists of services individually.
    Returns empty string if no differences, otherwise a dict keyed by protocol.
    """

    # Coerce NA to empty lists
    if not isinstance(left, (np.ndarray)):
        left = []
    if not isinstance(right, (np.ndarray)):
        right = []

    # Turn lists into dicts keyed by protocol (or (protocol, port) if needed)
    left_map  = {(s.get("protocol"), s.get("port")): s for s in left}
    right_map = {(s.get("protocol"), s.get("port")): s for s in right}

    diffs = {}

    # Services removed
    for k in left_map.keys() - right_map.keys():
        diffs[k] = "removed"

    # Services added
    for k in right_map.keys() - left_map.keys():
        diffs[k] = "added"

    # Services present in both → compare individually
    for k in left_map.keys() & right_map.keys():
        v1 = ujson.loads(left_map[k]["data"])
        v2 = ujson.loads(right_map[k]["data"])
        if c := compare_services(k[0], v1, v2):
            diffs[k] = c

    return diffs


def differences(src: pd.DataFrame, dst: pd.DataFrame) -> pd.DataFrame:
    merged = src.merge(dst, on="ip", how="outer", suffixes=("_left", "_right"))

    merged["ports_diff"] = merged.apply(
        lambda r: diff_ports(r["ports_left"], r["ports_right"]), axis=1 # type: ignore
    ) # type: ignore

    merged["services_diff"] = merged.apply(
        lambda r: diff_services(r["services_left"], r["services_right"]), axis=1
    )

    out = merged[[
        "ip",
        "ports_diff",
        "services_diff",
    ]]

    # Filter rows where ports_diff is not empty
    out = out[
        (out["ports_diff"].astype(bool)) |
        (out["services_diff"].astype(bool))
    ]

    return out.reset_index(drop=True)

def dump(df, path):
    records = df.to_dict(orient="records")
    with open(path, "+a") as f:
        for r in records:
            ujson.dump(r, f)
            f.write("\n")


def compare(repo: Repository, query: str, dst: str, fields: list[str], output: str="comparison.jsonl") -> Source:
    parser = make_parser()
    q = parser.to_sql(query)

    t, batches = repo.query_batch_n(q)

    con = repo.connect()
    info_b = new_info(fields)

    with tqdm(total=t, desc="compare") as pbar:
        for b in batches:
            ips = b.ip.tolist()

            srcdf = con.execute(info_b.make(ips)).df()
            dstdf = con.execute(info_b.make(ips, dst)).df()

            difs = differences(srcdf, dstdf)
            dump(difs, output)
            # some way to save this

            pbar.update(len(ips))

    source = new_source("comparison", output, "-")
    return source