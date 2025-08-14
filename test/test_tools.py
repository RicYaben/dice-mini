import os
import tempfile
from typing import Callable

from dice import repo, helpers, models

def make_test_sources() -> tuple[list[models.Source], Callable]:
    tmpdir = tempfile.TemporaryDirectory()
    fpath = os.path.join(tmpdir.name, "results.jsonl")
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(
            '{"ip":"2.2.2.2","port":4242,"data":{"test":{"status":"success","protocol":"test"}}}\r\n'+
            '{"ip":"1.1.1.1","port":4242,"data":{"test":{"status":"success","protocol":"test"}}}\r\n'
        )
    
    # takes a name of the source and a path to a file
    src = helpers.new_source("zgrab2", fpath)
    return ([src], tmpdir.cleanup)

def load_test_repository() -> repo.Repository:
        srcs, clean = make_test_sources()
        try:
            r = repo.load_repository(srcs)
            return r
        finally:
            clean()