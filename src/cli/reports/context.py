from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from cyclopts import Parameter

from dice.cli.helpers import token_converter
from dice.cli.models import SearchOptions


@dataclass
class AnnonimizerArgs:
    anonimize: Annotated[
        list[str] | None,
        Parameter(
            name=["--anonimize", "-a"],
            help="Comma-separated list of columns to anonymize",
            converter=token_converter(","),
        ),
    ] = None
    output: Annotated[
        Path | None,
        Parameter(
            name=["--output", "-o"],
            help="Output file path for anonimization mappings",
        ),
    ] = None
    remove: Annotated[
        list[str] | None,
        Parameter(
            name=["--remove", "-r"],
            help="Comma-separated list of columns to remove",
            converter=token_converter(","),
        ),
    ] = None


@dataclass
class ReportOptions:
    search: Annotated[SearchOptions | None, Parameter(name="*", group="Search")] = None
    anonimizer: Annotated[
        AnnonimizerArgs | None, Parameter(name="*", group="Annonimizer")
    ] = None


@dataclass
class ReportContext:
    search: SearchOptions
    anonimizer: AnnonimizerArgs


def make_context(opts: ReportOptions | None) -> ReportContext:
    if not opts:
        return ReportContext(
            search=SearchOptions(),
            anonimizer=AnnonimizerArgs(),
        )

    return ReportContext(
        search=opts.search if opts.search else SearchOptions(),
        anonimizer=opts.anonimizer if opts.anonimizer else AnnonimizerArgs(),
    )
