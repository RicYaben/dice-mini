import logging

from dice.sdk import Module
from dice.shared import Flags, flag
from dice.shared.models import Record
from dice.shared.query import query
from dice.shared.repository import CRepo


class ExampleFlags(Flags):
    eg: str = flag("-", "an example of flags")


def run(repo: CRepo, flags: ExampleFlags, logger: logging.Logger):
    kwargs = {
        "data.association.Msg.CalledAETitle": "ORTHANC",
        "protocol": "dicom",
    }
    q = query(
        Record,
        clauses=kwargs,
    )
    for r in repo.query(q):
        repo.label(r["id"], "label")


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
