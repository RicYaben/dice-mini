from dataclasses import dataclass
from typing import Any

from sqlalchemy import CTE, JSON, Select, func, select, type_coerce

from dice.shared.models import (
    Fingerprint,
    FingerprintLabel,
    Host,
    HostTag,
    Label,
    Record,
    Tag,
)


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
        if "all" in fields:
            return cls.all()

        return cls(
            ports="ports" in fields,
            services="services" in fields,
            tags="tags" in fields,
            labels="labels" in fields,
        )


@dataclass
class Report:
    ip: str
    prefix: str | None
    asn: str | None

    ports: list[int] | None = None
    services: list[dict[str, Any]] | None = None
    tags: list[str] | None = None
    labels: list[str] | None = None


class ReportBuilder:
    def __init__(self, fields: ReportFields):
        self.fields = fields

    def _fingerprints(self, hosts: list[str]) -> CTE:
        return (
            select(
                Fingerprint.host.label("ip"),
                Fingerprint.id.label("fingerprint_id"),
                Fingerprint.protocol,
                Record.port,
                Fingerprint.data,
            )
            .join(
                Record,
                Record.id == Fingerprint.record_id,
            )
            .where(Fingerprint.host.in_(hosts))
            .cte("fingerprints")
        )

    @staticmethod
    def _fingerprint_labels(fingerprints) -> CTE:
        return (
            select(
                fingerprints.c.fingerprint_id,
                func.json_group_array(func.distinct(Label.name)).label("labels"),
            )
            .select_from(fingerprints)
            .join(
                FingerprintLabel,
                FingerprintLabel.fingerprint_id == fingerprints.c.fingerprint_id,
            )
            .join(
                Label,
                Label.id == FingerprintLabel.label_id,
            )
            .group_by(fingerprints.c.fingerprint_id)
            .cte("fingerprint_labels")
        )

    def _fingerprint_info(self, fingerprints) -> CTE:
        columns = [fingerprints.c.ip]

        fingerprint_labels = None

        if self.fields.services:
            fingerprint_labels = self._fingerprint_labels(fingerprints)

        if self.fields.ports:
            ports = (
                func.json_group_array(func.distinct(fingerprints.c.port))
                .filter(fingerprints.c.port.is_not(None))
                .label("ports")
            )

            columns.append(ports)

        if self.fields.services:
            labels = func.coalesce(
                fingerprint_labels.c.labels,
                func.json_array(),
            )

            service = func.json_object(
                "protocol",
                fingerprints.c.protocol,
                "port",
                fingerprints.c.port,
                "data",
                func.json(fingerprints.c.data),
                "labels",
                labels,
            )

            services = func.json_group_array(func.json(service)).label("services")

            columns.append(services)

        stmt = select(*columns).select_from(fingerprints)

        if fingerprint_labels is not None:
            stmt = stmt.outerjoin(
                fingerprint_labels,
                fingerprint_labels.c.fingerprint_id == fingerprints.c.fingerprint_id,
            )

        return stmt.group_by(fingerprints.c.ip).cte("fingerprint_info")

    @staticmethod
    def _host_labels(fingerprints) -> CTE:
        return (
            select(
                fingerprints.c.ip,
                func.json_group_array(func.distinct(Label.name)).label("labels"),
            )
            .select_from(fingerprints)
            .join(
                FingerprintLabel,
                FingerprintLabel.fingerprint_id == fingerprints.c.fingerprint_id,
            )
            .join(
                Label,
                Label.id == FingerprintLabel.label_id,
            )
            .group_by(fingerprints.c.ip)
            .cte("host_labels")
        )

    @staticmethod
    def _tags(hosts: list[str]) -> CTE:
        return (
            select(
                HostTag.host.label("ip"),
                func.json_group_array(func.distinct(Tag.name)).label("tags"),
            )
            .select_from(HostTag)
            .join(
                Tag,
                Tag.id == HostTag.tag_id,
            )
            .where(HostTag.host.in_(hosts))
            .group_by(HostTag.host)
            .cte("host_tags")
        )

    @staticmethod
    def _json_array(column):
        return type_coerce(
            func.coalesce(
                column,
                func.json_array(),
            ),
            JSON,
        )

    def build(self, hosts: list[str]) -> Select:
        stmt = select(
            Host.ip,
            Host.prefix,
            Host.asn,
        ).where(Host.ip.in_(hosts))

        fingerprints = None

        if self.fields.ports or self.fields.services or self.fields.labels:
            fingerprints = self._fingerprints(hosts)

        if self.fields.ports or self.fields.services:
            fingerprint_info = self._fingerprint_info(fingerprints)

            if self.fields.ports:
                stmt = stmt.add_columns(
                    self._json_array(fingerprint_info.c.ports).label("ports")
                )

            if self.fields.services:
                stmt = stmt.add_columns(
                    self._json_array(fingerprint_info.c.services).label("services")
                )

            stmt = stmt.outerjoin(
                fingerprint_info,
                fingerprint_info.c.ip == Host.ip,
            )

        if self.fields.labels:
            host_labels = self._host_labels(fingerprints)

            stmt = stmt.add_columns(
                self._json_array(host_labels.c.labels).label("labels")
            )

            stmt = stmt.outerjoin(
                host_labels,
                host_labels.c.ip == Host.ip,
            )

        if self.fields.tags:
            tags = self._tags(hosts)

            stmt = stmt.add_columns(self._json_array(tags.c.tags).label("tags"))

            stmt = stmt.outerjoin(
                tags,
                tags.c.ip == Host.ip,
            )

        return stmt.order_by(Host.ip)

    @staticmethod
    def from_row(row) -> Report:
        return Report(
            ip=row.ip,
            prefix=row.prefix,
            asn=row.asn,
            ports=row.ports or [],
            services=row.services or [],
            tags=row.tags or [],
            labels=row.labels or [],
        )
