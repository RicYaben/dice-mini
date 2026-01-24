import pandas as pd
import glob

from typing import Generator
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
        
def jsonl_reader(p: Path, batch_size: int) -> Generator[pd.DataFrame, None, None]:
        # NOTE: engine pyarrow does not support chunking
        for c in pd.read_json(p, lines=True, dtype=True, convert_dates=False, chunksize=batch_size):
            yield c
            

def csv_reader(p: Path, batch_size: int) -> Generator[pd.DataFrame, None, None]:
    for c in pd.read_csv(p, chunksize=batch_size):
        yield c
        
def get_reader(ext: str):
    match ext:
        case ".jsonl":
            return jsonl_reader
        case ".csv":
            return csv_reader
        case _:
            raise Exception(f"usupported file extension: {ext}")
        
def file_loader(source_id: str, source_name: str, study: str, path: str, batch_size: int) -> Generator[pd.DataFrame, None, None]:
    norm = get_loader_normalizer(source_name)
    for p in walk(path):
        reader = get_reader(p.suffixes[0])
        for c in reader(p, batch_size):
            c["path"] = str(p)
            c["source_id"] = source_id
            c["source_name"] = source_name
            c["study"] = study
            c = norm(c)
            yield c
