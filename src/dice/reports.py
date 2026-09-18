from __future__ import annotations

from collections.abc import Generator, Sequence
from contextlib import ExitStack

from dice.internal.ast import make_parser
from dice.internal.reports import (
    Comparison,
    Report,
    ReportBuilder,
    ReportComparator,
    ReportFields,
)
from dice.shared.repository import Repository


def reports(
    repo: Repository, query: str | None, fields: list[str], bsize: int | None = None
) -> Generator[Report, None, None]:
    """Fetch reports from a repository."""

    rfields = ReportFields.from_fields(fields)
    rbuilder = ReportBuilder(rfields)

    if query:
        parser = make_parser()
        res = repo.search(parser.to_sql(query or ""))
        rbuilder.hosts([row["ip"] for row in res])

    stmt = rbuilder.build()
    for res in repo.search(stmt).batch(bsize=bsize):
        for row in res:
            yield Report.from_mappings(row)


def compare(
    repositories: Sequence[Repository],
    fields: list[str],
    query: str | None,
    bsize: int | None,
) -> Generator[Comparison, None, None]:
    if not repositories:
        raise ValueError("At least one repository is required")

    parser = make_parser()

    rfields = ReportFields.from_fields(fields)
    rbuilder = ReportBuilder(rfields)
    comparator = ReportComparator()

    base = repositories[0].search(parser.to_sql(query or ""))

    with ExitStack() as stack:
        connections = [
            stack.enter_context(repository.connect()) for repository in repositories
        ]

        for batch in base.batch(bsize):
            ips = [row["ip"] for row in batch]

            rbuilder.hosts(ips)
            statement = rbuilder.build()

            repository_reports = [
                {
                    report.ip: report
                    for report in (
                        Report.from_mappings(row)
                        for row in connection.execute(statement).mappings()
                    )
                }
                for connection in connections
            ]

            for ip in ips:
                reports = [
                    repository_reports[index].get(ip)
                    for index in range(len(repositories))
                ]

                yield comparator.compare(reports)
