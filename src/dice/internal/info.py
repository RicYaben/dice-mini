from sqlalchemy import select, func
from sqlalchemy.sql import Select

from dice.shared.models import Host, Fingerprint, Label, FingerprintLabel, HostTag, Tag


class InfoQueryBuilder:
    def __init__(self, fields: list[str]):
        fields = list(dict.fromkeys(fields))

        if "all" in fields:
            fields = ["ports", "services", "tags", "labels"]

        self.fields = set(fields)

    def make(self, hosts: list[str]) -> Select:
        hosts_cte = (
            select(Host.ip, Host.prefix, Host.asn) # type: ignore
            .where(Host.ip.in_(hosts)) # type: ignore
            .cte("h")
        )

        stmt = select(hosts_cte.c.ip, hosts_cte.c.prefix, hosts_cte.c.asn)

        if "ports" in self.fields:
            ports_sub = (
                select(
                    Fingerprint.host.label("ip"),
                    func.group_concat(func.distinct(Fingerprint.port)).label("ports"),
                )
                .where(Fingerprint.host.in_(hosts))
                .group_by(Fingerprint.host)
                .subquery()
            )

            stmt = stmt.add_columns(ports_sub.c.ports)
            stmt = stmt.outerjoin(ports_sub, ports_sub.c.ip == hosts_cte.c.ip)

        if "services" in self.fields:
            labels_sub = (
                select(
                    FingerprintLabel.fingerprint_id,
                    func.group_concat(Label.name, ",").label("labels"),
                ) # type: ignore
                .join(Label, Label.id == FingerprintLabel.label_id)
                .group_by(FingerprintLabel.fingerprint_id)
                .subquery()
            )

            fp_sub = (
                select(
                    Fingerprint.host.label("ip"),
                    Fingerprint.id,
                    Fingerprint.protocol,
                    Fingerprint.port,
                    Fingerprint.data,
                    labels_sub.c.labels,
                )
                .outerjoin(labels_sub, labels_sub.c.fingerprint_id == Fingerprint.id)
                .where(Fingerprint.host.in_(hosts))
                .subquery()
            )

            services_sub = (
                select(
                    fp_sub.c.ip,
                    func.group_concat(
                        func.json_object(
                            "protocol",
                            fp_sub.c.protocol,
                            "port",
                            fp_sub.c.port,
                            "data",
                            fp_sub.c.data,
                            "labels",
                            func.coalesce(fp_sub.c.labels, ""),
                        )
                    ).label("services"),
                )
                .group_by(fp_sub.c.ip)
                .subquery()
            )

            stmt = stmt.add_columns(services_sub.c.services)
            stmt = stmt.outerjoin(services_sub, services_sub.c.ip == hosts_cte.c.ip)

        if "tags" in self.fields:
            tags_sub = (
                select(
                    HostTag.host.label("ip"),
                    func.group_concat(Tag.name, ",").label("tags"),
                )
                .join(Tag, Tag.id == HostTag.tag_id)
                .where(HostTag.host.in_(hosts))
                .group_by(HostTag.host)
                .subquery()
            )

            stmt = stmt.add_columns(tags_sub.c.tags)
            stmt = stmt.outerjoin(tags_sub, tags_sub.c.ip == hosts_cte.c.ip)

        stmt = stmt.select_from(hosts_cte).order_by(hosts_cte.c.ip)
        return stmt


def new_info(fields: list[str]) -> InfoQueryBuilder:
    return InfoQueryBuilder(fields)
