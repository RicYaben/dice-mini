from collections.abc import Sequence
from typing import Any

from sqlalchemy import CTE, JSON, Select, func, type_coerce
from sqlmodel import select

from dice.shared.models import (
    Fingerprint,
    FingerprintLabel,
    Host,
    HostTag,
    Label,
    Record,
    Tag,
)

from .models import ReportFields


class ReportBuilder:
    """Build SQL statements for reports."""

    def __init__(
        self,
        fields: ReportFields | None = None,
    ):
        self._fields = fields or ReportFields()
        self._hosts: list[str] | None = None

    @classmethod
    def from_fields(
        cls,
        fields: Sequence[str],
    ) -> "ReportBuilder":
        return cls(ReportFields.from_fields(list(fields)))

    def ports(
        self,
        enabled: bool = True,
    ) -> "ReportBuilder":
        self._fields = ReportFields(
            ports=enabled,
            services=self._fields.services,
            tags=self._fields.tags,
            labels=self._fields.labels,
        )
        return self

    def services(
        self,
        enabled: bool = True,
    ) -> "ReportBuilder":
        self._fields = ReportFields(
            ports=self._fields.ports,
            services=enabled,
            tags=self._fields.tags,
            labels=self._fields.labels,
        )
        return self

    def tags(
        self,
        enabled: bool = True,
    ) -> "ReportBuilder":
        self._fields = ReportFields(
            ports=self._fields.ports,
            services=self._fields.services,
            tags=enabled,
            labels=self._fields.labels,
        )
        return self

    def labels(
        self,
        enabled: bool = True,
    ) -> "ReportBuilder":
        self._fields = ReportFields(
            ports=self._fields.ports,
            services=self._fields.services,
            tags=self._fields.tags,
            labels=enabled,
        )
        return self

    def all(
        self,
        enabled: bool = True,
    ) -> "ReportBuilder":
        self._fields = ReportFields.all() if enabled else ReportFields()
        return self

    def hosts(
        self,
        hosts: Sequence[str],
    ) -> "ReportBuilder":
        self._hosts = list(hosts)
        return self

    def build(self) -> Select:
        """Build a statement for the configured hosts."""

        if self._hosts is None:
            raise ValueError("No hosts configured. Call hosts() before build().")

        return self._build(self._hosts)

    def _build(
        self,
        hosts: Sequence[str],
    ) -> Select:
        statement = select(
            Host.ip,
            Host.prefix,
            Host.asn,
        ).where(Host.ip.in_(hosts))

        needs_fingerprints = (
            self._fields.ports or self._fields.services or self._fields.labels
        )

        fingerprints = self._fingerprints(hosts) if needs_fingerprints else None

        if self._fields.ports or self._fields.services:
            assert fingerprints is not None

            fingerprint_info = self._fingerprint_info(fingerprints)

            if self._fields.ports:
                statement = statement.add_columns(
                    self._json_array(fingerprint_info.c.ports).label("ports")
                )

            if self._fields.services:
                statement = statement.add_columns(
                    self._json_array(fingerprint_info.c.services).label("services")
                )

            statement = statement.outerjoin(
                fingerprint_info,
                fingerprint_info.c.ip == Host.ip,
            )

        if self._fields.labels:
            assert fingerprints is not None

            host_labels = self._host_labels(fingerprints)

            statement = statement.add_columns(
                self._json_array(host_labels.c.labels).label("labels")
            )

            statement = statement.outerjoin(
                host_labels,
                host_labels.c.ip == Host.ip,
            )

        if self._fields.tags:
            tags = self._tags(hosts)

            statement = statement.add_columns(
                self._json_array(tags.c.tags).label("tags")
            )

            statement = statement.outerjoin(
                tags,
                tags.c.ip == Host.ip,
            )

        return statement.order_by(Host.ip)

    def _fingerprints(
        self,
        hosts: Sequence[str],
    ) -> CTE:
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
    def _fingerprint_labels(
        fingerprints: CTE,
    ) -> CTE:
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

    def _fingerprint_info(
        self,
        fingerprints: CTE,
    ) -> CTE:
        columns = [fingerprints.c.ip]

        fingerprint_labels = None

        if self._fields.services:
            fingerprint_labels = self._fingerprint_labels(fingerprints)

        if self._fields.ports:
            ports = (
                func.json_group_array(func.distinct(fingerprints.c.port))
                .filter(fingerprints.c.port.is_not(None))
                .label("ports")
            )

            columns.append(ports)

        if self._fields.services:
            assert fingerprint_labels is not None

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

        statement = select(*columns).select_from(fingerprints)

        if fingerprint_labels is not None:
            statement = statement.outerjoin(
                fingerprint_labels,
                fingerprint_labels.c.fingerprint_id == fingerprints.c.fingerprint_id,
            )

        return statement.group_by(fingerprints.c.ip).cte("fingerprint_info")

    @staticmethod
    def _host_labels(
        fingerprints: CTE,
    ) -> CTE:
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
    def _tags(
        hosts: Sequence[str],
    ) -> CTE:
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
    def _json_array(
        column: Any,
    ) -> Any:
        return type_coerce(
            func.coalesce(
                column,
                func.json_array(),
            ),
            JSON,
        )
