from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from cyclopts import Parameter

from dice.cli.config.helpers import token_converter


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
class SearchArgs:
    query: Annotated[
        str | None, Parameter(name=["--query", "-q"], help="Search query")
    ] = None
    include: Annotated[
        list[str] | None,
        Parameter(
            name=["--include", "-i"],
            help="Comma-separated list of columns to include",
            converter=token_converter(","),
        ),
    ] = None
    exclude: Annotated[
        list[str] | None,
        Parameter(
            name=["--exclude", "-e"],
            help="Comma-separated list of columns to exclude",
            converter=token_converter(","),
        ),
    ] = None
    limit: Annotated[
        int | None,
        Parameter(
            name=["--limit", "-l"],
            help="Limit the number of results",
        ),
    ] = None
    offset: Annotated[
        int | None,
        Parameter(
            name=["--offset", "-o"],
            help="Offset the results",
        ),
    ] = None


@dataclass
class SearchOptions:
    search: Annotated[SearchArgs | None, Parameter(name="*", group="Search")] = None
    anonimizer: Annotated[
        AnnonimizerArgs | None, Parameter(name="*", group="Annonimizer")
    ] = None


@dataclass
class SearchContext:
    search: SearchArgs
    anonimizer: AnnonimizerArgs


def make_context(opts: SearchOptions | None) -> SearchContext:
    if not opts:
        return SearchContext(
            search=SearchArgs(),
            anonimizer=AnnonimizerArgs(),
        )

    return SearchContext(
        search=opts.search if opts.search else SearchArgs(),
        anonimizer=opts.anonimizer if opts.anonimizer else AnnonimizerArgs(),
    )
