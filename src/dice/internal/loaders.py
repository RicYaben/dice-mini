from collections.abc import Callable, Generator, Iterator
from pathlib import Path
from typing import Any

import pandas as pd


class UnsupportedFileExtensionError(Exception):
    def __init__(self, ext: str):
        self.ext = ext
        super().__init__(f"unsupported file extension: {ext}")


def walk(p: str | Path) -> Iterator[Path]:
    path = Path(p)
    if path.is_file():
        yield path
        return

    if path.is_dir():
        yield from (f for f in path.rglob("*") if f.is_file())
        return

    # Treat non-existent paths as glob patterns.
    yield from (f for f in path.parent.glob(path.name) if f.is_file())


def extract_protocol_data(d: dict) -> tuple[str, dict]:
    try:
        first_obj: dict[str, Any] = next(iter(d.values()))
    except StopIteration:
        return "", {}

    protocol: str = first_obj.get("protocol", "-")
    if "result" not in first_obj:
        return protocol, first_obj

    first_obj.update(first_obj["result"])
    del first_obj["result"]

    return protocol, first_obj


def zgrab2_loader_normalizer(df: pd.DataFrame) -> pd.DataFrame:
    df[["protocol", "data"]] = df["data"].apply(
        lambda raw: pd.Series(extract_protocol_data(raw))
    )
    df = df.rename({"ip": "host"}, axis=1)

    if "port" not in df.columns:
        df["port"] = -1
    return df


def get_loader_normalizer(source: str) -> Callable[[pd.DataFrame], pd.DataFrame]:
    match source:
        case "zgrab2":
            return zgrab2_loader_normalizer
        case _:
            return lambda x: x


def jsonl_reader(p: Path, batch_size: int) -> Generator[pd.DataFrame, None, None]:
    reader = pd.read_json(
        p,
        lines=True,
        convert_dates=False,
        chunksize=batch_size,
        encoding="utf-8",
        encoding_errors="ignore",
    )

    yield from reader


def csv_reader(p: Path, batch_size: int) -> Generator[pd.DataFrame, None, None]:
    reader = pd.read_csv(p, chunksize=batch_size)
    yield from reader


def get_reader(ext: str):
    match ext:
        case ".jsonl":
            return jsonl_reader
        case ".csv":
            return csv_reader
        case _:
            raise UnsupportedFileExtensionError(ext)


def read_resource(
    resource_id: int, fpath: Path, batch_size: int
) -> Generator[pd.DataFrame, None, None]:
    reader = get_reader(fpath.suffixes[0])
    for c in reader(fpath, batch_size):
        c["resource_id"] = resource_id
        yield c
