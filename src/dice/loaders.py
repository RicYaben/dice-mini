import pandas as pd
import glob

from typing import Generator, Iterable
from collections.abc import Callable
from pathlib import Path

type Loader = Callable[[str, str, str, str, int], Generator[pd.DataFrame, None, None]]

def walk(p: str):
    """
    Iterate over a list of paths that can be:
      - Directories → yields all files in them recursively
      - Glob patterns → yields matches
      - File paths → yields the file if it exists
    """
    path = Path(p)
    
    if "*" in p or "?" in p or "[" in p:
        # Glob pattern
        for match in glob.iglob(p, recursive=True):
            match_path = Path(match)
            if match_path.is_file():
                yield match_path
    elif path.is_dir():
        # Directory → recursively yield files
        for f in path.rglob("*"):
            if f.is_file():
                yield f
    elif path.is_file():
        # Direct file path
        yield path

def extract_protocol_data(d: dict) -> tuple[str, dict]:
    try:
        first_obj = list(d.values())[0]
        protocol = first_obj.get("protocol")
        return protocol, first_obj
    except Exception:
        return "", {}

def zgrab2_loader_normalizer(df: pd.DataFrame) -> pd.DataFrame:
    df[["protocol","data"]] = df["data"].apply(lambda raw: pd.Series(extract_protocol_data(raw)))
    return df

def get_loader_normalizer(source: str) -> Callable[[pd.DataFrame], pd.DataFrame]:
    match source:
        case "zgrab2":
            return zgrab2_loader_normalizer
        case _:
            return lambda x: x

def jsonl_loader(source_id: str, source_name: str, study: str, path: str, batch_size: int):
    norm = get_loader_normalizer(source_name)
    for p in walk(path):
        # NOTE: engine pyarrow does not support chunking
        for chunk in pd.read_json(p, lines=True, dtype=True, convert_dates=False, chunksize=batch_size):
            chunk["path"] = str(p)
            chunk["source_id"] = source_id
            chunk["source_name"] = source_name
            chunk["study"] = study
            chunk = norm(chunk)
            yield chunk

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