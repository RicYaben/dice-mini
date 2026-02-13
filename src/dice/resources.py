from itertools import chain
from typing import Generator, Optional
from sqlalchemy import Connection
from sqlmodel import Session
from tqdm import tqdm

from dice.config import DEFAULT_BSIZE
from dice.database import get_or_create
from dice.loaders import read_resource
from dice.models import Resource, Cursor

import ujson
import pandas as pd
import logging
import os

from dice.repo import Repository

logger = logging.getLogger(__name__)


class Sourcerer:
    """Something to load sources"""
    _gen: Optional[Generator[pd.DataFrame, None, None]] = None
    _peek: Optional[pd.DataFrame] = None
    _peeked: bool = False

    _oc: list[str] = []
    _ic: list[str] = []

    def __init__(self, rsrc: Resource, cursor: Cursor, resume: bool, bsize: int) -> None:
        self.resume = resume
        self.bsize = bsize
        self.rsrc = rsrc
        self.cursor = cursor
    
    @property
    def peek(self) -> pd.DataFrame | None:
        if not self._gen:
            self.load()

        if self._peeked:
            return self._peek
        
        try:
            assert isinstance(self._gen, Generator)
            p = next(self._gen)
            self._peek = p
        except StopIteration:
            pass

        self._peeked = True
        return self._peek
    
    @property
    def columns(self)  -> tuple[list[str], list[str]]:
        if self._oc or self._ic:
            return (self._oc, self._ic)
        
        if self.empty():
            raise ValueError(f"unable to get columns: empty source {self.rsrc.fpath}") 

        p = self.peek
        assert isinstance(p, pd.DataFrame)

        n = min(1000, len(p))
        logger.debug(f"polling source with {n}/{len(p)}")
        s = p.sample(n)

        # object cols
        oc = []
        for col in p.select_dtypes(include=["object"]).columns:
            if s[col].apply(lambda x: isinstance(x, (dict, list))).any():
                oc.append(col)

        # numeric cols
        ic = list(p.select_dtypes(include=["number"]).columns)

        self._oc = oc
        self._ic = ic
        return (oc, ic)
    
    def exists(self) -> bool:
        return os.path.exists(self.rsrc.fpath)
    
    def load(self) -> Generator[pd.DataFrame, None, None]:
        if self._gen:
            return self._gen
        
        data = read_resource(self.rsrc.id.hex, self.rsrc.fpath, self.bsize)
        for _ in range(self.cursor.index):
            next(data, None)

        self._gen = data
        return data

    def reset(self):
        self._gen = None
        self._peek = None
        self._peeked = False

    def format_columns(self, df: pd.DataFrame, oc, ic: list[str]) -> pd.DataFrame:
        # convert to string dict and list cols
        for col in oc:
            df[col] = df[col].map(
                lambda x: ujson.dumps(x) if isinstance(x, (dict, list)) else x
            )

        # convert to int64 numeric cols
        for col in ic:
            df[col] = pd.to_numeric(df[col], errors="coerce", dtype_backend="pyarrow", downcast="float")

        df["resource_id"] = [self.rsrc.id.hex for _ in range(len(df))]
        return df

    def cast(self, con: Connection) -> Generator[pd.DataFrame, None, None]:
        if not self.resume or self.cursor.index < 0:
            # we change the cursor to the beggining
            self.cursor.index = 0
            # delete all the records stored from this resource to avoid dupes
            self.rsrc.flush_records(con)

        data = self.load()
        p = self.peek
        assert isinstance(p, pd.DataFrame)

        oc, ic = self.columns
        for c in chain([p], data):
            fmt = self.format_columns(c, oc, ic)
            yield fmt

            self.cursor.index += 1

            with Session(con) as s:
                s.add(self.rsrc)
                s.commit()

        with Session(con) as s:
            self.cursor.index = -1
            s.add(self.rsrc)
            s.commit()

        # reset the peek and generator
        self.reset()

    def empty(self) -> bool:
        p = self.peek
        return p is None or p.empty
    
    def check(self):
        if not self.exists():
            raise ValueError(f"source not found: {self.rsrc.fpath}")
        if self.empty():
            raise ValueError(f"empty resource: {self.rsrc.fpath}")


def new_sourcerer(resource: Resource, cursor: Cursor, resume: bool, bsize: int) -> Sourcerer:
    return Sourcerer(resource, cursor, resume, bsize)


def add_resource(repo: Repository, name: str, source: str, fpath: str, resume: bool = True, bsize: int = DEFAULT_BSIZE):
        logger.info(f"adding resource from {fpath} ({bsize}/b)")

        # load the resource or create it with its cursor
        with repo.session() as s:
            res, _ = get_or_create(s, Resource, fpath=fpath, source_id=source)
            cursor, _ = get_or_create(s, Cursor, resource_id=res.id.hex)
            res.cursor = cursor
            s.commit()
            s.refresh(res)

            sourcerer = new_sourcerer(res, cursor, resume, bsize)
            sourcerer.load()

        with repo.connect() as con:
            gen = sourcerer.cast(con)
            for c in tqdm(gen):
                c.to_sql(f"{name}_records", con, if_exists="append", index=False)
