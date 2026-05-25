from dice.sdk import Module
from dice.shared import Flags, flag
from dice.shared.models import Record
from dice.shared.repository import CRepo
from dice.experimental import query

import logging

class ExampleFlags(Flags):
    eg: str = flag("-", "an example of flags")

def run(repo: CRepo, flags: ExampleFlags, logger: logging.Logger):
    q = query(Record, protocol="dicom", data__result__scheme="tcp")
    for r in repo.query(q):
        logger.info(r)
        #repo.label(r["id"], "label")

def example_classifier() -> Module:
    return (
        Module(
            "c",
            "example",
            flags=ExampleFlags,
            run_fn=run,
        )
        .add_label("label", "desc")
        .add_tag("tag", "desc")
    )
