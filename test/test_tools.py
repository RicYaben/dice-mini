import os
from tempfile import TemporaryDirectory
from sqlmodel import Session

from dice.internal.database import get_or_create
from dice.internal.loaders import walk
from dice.internal.models import Source
from dice.internal.repository import Repository
from dice.cli.tools import load_repository

def make_test_zgrab2_source(s: Session, dir: str) -> tuple[Source, str]:
    fpath = os.path.join(dir, "results.jsonl")
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(
            '{"ip":"2.2.2.2","port":0,"data":{"test":{"status":"success","protocol":"test"}}}\r\n'+
            '{"ip":"1.1.1.1","port":0,"data":{"test":{"status":"success","protocol":"test"}}}\r\n'
        )

    src, _ = get_or_create(s, Source, name="zgrab2")
    return (src, fpath)

def load_test_repository() -> Repository:
        dir = TemporaryDirectory()
        try:
            repo = load_repository()
            with repo.session() as s:
                fpath, fpath = make_test_zgrab2_source(s, dir.name)

            for p in walk(fpath):
                add_resource(repo, src.name, src.id, str(p), resume=false, bsize=batch) # type: ignore
            return repo
        finally:
            dir.cleanup()

def summary(repo: Repository) -> dict:
    with repo.connect() as con:
         return {}