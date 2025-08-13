import pandas as pd
import glob

from collections.abc import Callable
from pathlib import Path

type Loader = Callable[[str, str, list[str]], pd.DataFrame]

def walk(paths):
    """
    Iterate over a list of paths that can be:
      - Directories → yields all files in them recursively
      - Glob patterns → yields matches
      - File paths → yields the file if it exists
    """
    for p in paths:
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

def jsonl_loader(id: str, name: str, paths: list[str]) -> pd.DataFrame:
    dfs = []
    for p in walk(paths):
        df = pd.read_json(p, lines=True).assign(path=str(p))
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True).assign(id=id, name=name)