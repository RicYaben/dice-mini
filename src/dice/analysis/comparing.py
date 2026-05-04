from difflib import ndiff
from sqlalchemy import MetaData
from tqdm import tqdm

from dice.ast import make_parser
from dice.info import new_info
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
    return "\n".join(ndiff(str(old).splitlines(), str(new).splitlines()))


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
    except Exception:
        pass

    return v1 == v2


def changes(src: dict, dst: dict, fields: list[str] | None = None) -> dict[str, str]:
    "returns differences between fingerprints"
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

            la = list(filter(ca100, left["asdus"]))
            ra = list(filter(ca100, right["asdus"]))
            if not is_equal(la, ra):
                return {"asdus": fmt_dif(la, ra)}
            return {}

        case "fox":
            fields = left.keys() - [
                "timestamp",
                "result",
                "error",
                "status",
                "is_fox",
                "probe_status",
                "protocol",
            ]
            return changes(left, right, fields)
        case _:
            return changes(left, right)


def _normalize_services(x):
    """Ensure we always return list[dict]."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []

    # string → parse JSON
    if isinstance(x, str):
        try:
            x = ujson.loads(x)
        except Exception:
            return []

    # single dict → wrap
    if isinstance(x, dict):
        return [x]

    # already list
    if isinstance(x, list):
        return x

    return []


def _parse_data(s):
    """Safely parse the nested 'data' field."""
    if not isinstance(s, dict):
        return {}

    d = s.get("data")
    if isinstance(d, str):
        try:
            return ujson.loads(d)
        except Exception:
            return d
    return d

def diff_services(left, right):
    """
    Compare two services collections.
    Returns {} if no differences.
    """

    left = _normalize_services(left)
    right = _normalize_services(right)

    # map by (protocol, port)
    left_map = {
        (s.get("protocol"), s.get("port")): s
        for s in left if isinstance(s, dict)
    }
    right_map = {
        (s.get("protocol"), s.get("port")): s
        for s in right if isinstance(s, dict)
    }

    diffs = {}

    # removed
    for k in left_map.keys() - right_map.keys():
        diffs[k] = "removed"

    # added
    for k in right_map.keys() - left_map.keys():
        diffs[k] = "added"

    # compare common
    for k in left_map.keys() & right_map.keys():
        v1 = _parse_data(left_map[k])
        v2 = _parse_data(right_map[k])

        if v1 != v2:
            diffs[k] = compare_services(k[0], v1, v2) # type: ignore

    return diffs


def differences(src: pd.DataFrame, dst: pd.DataFrame) -> pd.DataFrame:
    merged = src.merge(dst, on="ip", how="outer", suffixes=("_left", "_right"))

    merged["ports_diff"] = merged.apply(
        lambda r: diff_ports(r["ports_left"], r["ports_right"]), axis=1
    )

    merged["services_diff"] = merged.apply(
        lambda r: diff_services(r["services_left"], r["services_right"]), axis=1
    )

    out = merged[
        [
            "ip",
            "ports_diff",
            "services_diff",
        ]
    ]

    # Filter rows where ports_diff is not empty
    out = out[(out["ports_diff"].astype(bool)) | (out["services_diff"].astype(bool))]

    return out.reset_index(drop=True)


def dump(df, path):
    records = df.to_dict(orient="records")
    with open(path, "+a") as f:
        for r in records:
            ujson.dump(r, f)
            f.write("\n")


def compare(
    r1: Repository,
    r2: Repository,
    query: str,
    fields: list[str],
    output: str = "comparison.jsonl",
) -> None:
    parser = make_parser()
    q = parser.to_sql(query)

    t, gen = r1.query(q)
    info_b = new_info(fields)

    c1 = r1.connect()
    c2 = r2.connect()

    meta = MetaData()
    meta.reflect(bind=c1)

    with tqdm(total=t, desc="compare") as pbar:
        for df in gen:
            ips = df.ip.tolist()

            q = info_b.make(ips, meta.tables)

            src_df = pd.read_sql(q, c1)
            dst_df = pd.read_sql(q, c2)

            difs = differences(src_df, dst_df)
            dump(difs, output)

            pbar.update(len(ips))
