from dataclasses import asdict, dataclass, field
from typing import Any

from sqlalchemy import RowMapping


@dataclass(frozen=True)
class ReportFields:
    ports: bool = False
    services: bool = False
    tags: bool = False
    labels: bool = False

    @classmethod
    def all(cls) -> "ReportFields":
        return cls(
            ports=True,
            services=True,
            tags=True,
            labels=True,
        )

    @classmethod
    def from_fields(cls, fields: list[str]) -> "ReportFields":
        if not fields or "all" in fields:
            return cls.all()

        return cls(
            ports="ports" in fields,
            services="services" in fields,
            tags="tags" in fields,
            labels="labels" in fields,
        )


@dataclass(frozen=True)
class Report:
    ip: str
    prefix: str | None
    asn: str | None

    ports: list[int] = field(default_factory=list)
    services: list[dict[str, Any]] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)

    @classmethod
    def from_mappings(cls, row: RowMapping) -> "Report":
        """Convert a database row into a report."""
        return Report(
            ip=row["ip"],
            prefix=row["prefix"],
            asn=row["asn"],
            ports=row.get("ports") or [],
            services=row.get("services") or [],
            tags=row.get("tags") or [],
            labels=row.get("labels") or [],
        )


@dataclass(frozen=True)
class ReportComparison:
    baseline: Report | None
    current: Report | None
    changes: dict[str, Any]

    @property
    def changed(self) -> bool:
        return bool(self.changes)


@dataclass(frozen=True)
class Comparison:
    ip: str
    reports: tuple[Report | None, ...]
    comparisons: tuple[ReportComparison, ...]

    @property
    def changed(self) -> bool:
        return any(comparison.changed for comparison in self.comparisons)

    @property
    def present(self) -> tuple[Report, ...]:
        return tuple(report for report in self.reports if report is not None)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
