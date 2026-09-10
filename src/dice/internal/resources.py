import logging
from collections.abc import Generator
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path

import pandas as pd
import ujson
from sqlalchemy import Connection
from sqlmodel import Session, col, select
from tqdm import tqdm

from dice.shared.models import Cursor, Record, Resource, Source

from .database import get_or_create
from .loaders import get_loader_normalizer, read_resource
from .repository import Repository

logger = logging.getLogger(__name__)


class ResourceNotFoundError(Exception):
    def __init__(self, res_id: int) -> None:
        self.res_id = res_id
        super().__init__(f"resource not found: {res_id}")


def load_resource(
    s: Session,
    res_id: int,
) -> tuple[Resource, Cursor, Source]:
    stmt = (
        select(Resource, Cursor, Source)
        .join(Cursor, col(Cursor.resource_id) == col(Resource.id))
        .join(Source, col(Source.id) == col(Resource.source_id))
        .where(col(Resource.id) == res_id)
    )

    if row := s.exec(stmt).first():
        return row
    raise ValueError(f"resource not found: {res_id}")


@dataclass
class Sourcerer:
    """Something to load sources"""

    res_id: int
    resume: bool
    bsize: int

    _gen: Generator[pd.DataFrame, None, None] | None = None
    _peek: pd.DataFrame | None = None
    _peeked: bool = False

    _ic: list[str] = field(default_factory=list)

    @property
    def peek(self) -> pd.DataFrame | None:
        if not self._gen:
            raise ResourceNotFoundError(self.res_id)

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

    # TODO: this may not be necessary anymore
    @property
    def columns(self) -> list[str]:
        if self._ic:
            return self._ic

        if self.empty():
            raise ValueError("unable to get columns: empty source")

        p = self.peek
        assert isinstance(p, pd.DataFrame)

        n = min(1000, len(p))
        logger.debug(f"polling source with {n}/{len(p)}")
        p.sample(n)

        # numeric cols
        ic = list(p.select_dtypes(include=["number"]).columns)

        self._ic = ic
        return ic

    def load(self, fpath: Path, i: int = 0) -> None:
        if self._gen:
            return

        gen = read_resource(self.res_id, fpath, self.bsize)
        for _ in range(i):
            next(gen, None)
        self._gen = gen

    def reset(self):
        self._gen = None
        self._peek = None
        self._peeked = False

    def format_columns(
        self, df: pd.DataFrame, res_id: int, ic: list[str]
    ) -> pd.DataFrame:
        # convert to int64 numeric cols
        for col in ic:
            df[col] = pd.to_numeric(
                df[col], errors="coerce", dtype_backend="pyarrow", downcast="float"
            )

        df["resource_id"] = res_id
        return df

    def cast(self, con: Connection) -> Generator[pd.DataFrame, None, None]:
        with Session(con) as s:
            res, cursor, src = load_resource(s, self.res_id)

            if not self.resume or cursor.idx < 0:
                # we change the cursor to the beggining
                cursor.idx = 0
                # delete all the records stored from this resource to avoid dupes
                res.flush_records(con)

            fpath = Path(res.fpath)
            self.load(fpath, cursor.idx)
            p = self.peek
            assert isinstance(p, pd.DataFrame)
            assert self._gen

            ic = self.columns
            norm = get_loader_normalizer(src.name)
            for df in chain([p], self._gen):
                ret = norm(df)
                fmt = self.format_columns(ret, self.res_id, ic)
                yield fmt
                cursor.update(s)

            cursor.done(s)
            # reset the peek and generator
            self.reset()

    def empty(self) -> bool:
        p = self.peek
        return p is None or p.empty

    def check(self, fpath: Path):
        if not fpath.exists():
            raise ValueError(f"source not found: {fpath}")
        if self.empty():
            raise ValueError(f"empty resource: {fpath}")


def new_resourcerer(res_id: int, resume: bool, bsize: int) -> Sourcerer:
    return Sourcerer(res_id, resume, bsize)


def add_resource(
    repo: Repository,
    source: Source,
    fpath: Path,
    bsize: int,
    resume: bool = True,
):
    logger.info(f"adding resource from {fpath} ({bsize}/b)")

    # load the resource or create it with its cursor
    with repo.session() as s:
        res, _ = get_or_create(s, Resource, fpath=fpath, source_id=source.id)
        cursor, _ = get_or_create(s, Cursor, resource_id=res.id)
        res.cursor = cursor
        s.commit()
        s.refresh(res)

        sourcerer = new_resourcerer(res.id, resume, bsize)

    with repo.connect() as con:
        gen = sourcerer.cast(con)
        for c in tqdm(gen):
            records = []
            for _, row in c.iterrows():
                data = row.get("data", {})
                assert isinstance(data, dict)

                records.append(
                    Record(
                        source=source.name,
                        resource_id=res.id,
                        host=row.get("host", None),
                        data=data,
                        port=row.get("port", None),
                        protocol=row.get("protocol", None),
                    )
                )
            repo.insert(records, con=con)
