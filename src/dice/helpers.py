import ujson
import pandas as pd

from typing import Any, Callable, Generator, Iterable

from dice.config import DATA_PREFIX
from dice.loaders import Loader

def normalize_data(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    # cannot parse
    if not df.iloc[0].get("data", None):
        return df

    parsed = df["data"].map(ujson.loads)
    rdf = pd.json_normalize(parsed.tolist(), max_level=0).add_prefix(prefix)
    norm = pd.concat(
        [df.drop(columns=["data"]).reset_index(drop=True), rdf.reset_index(drop=True)],
        axis=1,
    )
    return norm


def normalize_zgrab2_records(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    parsed = df["data"].apply(ujson.loads)

    # Flatten the 'result' dict
    rdf = pd.json_normalize(parsed.tolist(), max_level=1)
    rdf.columns = rdf.columns.str.removeprefix("result.")
    rdf = rdf.add_prefix(prefix)

    # Concatenate original df (without 'data') and flattened result columns
    norm = pd.concat(
        [df.drop(columns=["data"]).reset_index(drop=True), rdf.reset_index(drop=True)],
        axis=1,
    )

    return norm


def get_normalizer(src: str) -> Callable[[pd.DataFrame], pd.DataFrame]:
    match src:
        case "zgrab2":
            def ret(df: pd.DataFrame):
                return normalize_zgrab2_records(df, DATA_PREFIX)
            return ret
        case _:
            def ret(df: pd.DataFrame):
                return normalize_data(df, DATA_PREFIX)
            return ret


def normalize_fingerprints(df: pd.DataFrame) -> pd.DataFrame:
    return normalize_data(df, DATA_PREFIX)

def get_record_field(r, field: str, default: Any=None, prefix: str="data_") -> Any:
    v = r.get(prefix+field, default)

    if isinstance(v, (list, tuple)):
        return default if len(v) == 0 else v
    
    return v if not pd.isna(v) else default

def record_to_dict(r, prefix: str="data_") -> dict:
    d = r.to_dict()
    d = {k[len(prefix):]: v for k, v in d.items() if k.startswith(prefix)}
    return d

def with_records(records: Iterable[dict], chunk_size: int = 5_000) -> Loader:
    def load(*args, **kwargs) -> Generator[pd.DataFrame, None, None]:
        batch = []
        for rec in records:
            batch.append(rec)
            if len(batch) >= chunk_size:
                yield pd.DataFrame(batch)
                batch.clear()

        # Yield remaining records
        if batch:
            yield pd.DataFrame(batch)
    return load