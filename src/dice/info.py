from sqlalchemy import select, func
from sqlalchemy.sql import Select


class InfoQueryBuilder:
    """
    Safe SQLAlchemy query builder (SQLite compatible)
    """

    def __init__(self, fields: list[str]):
        fields = list(dict.fromkeys(fields))

        if "all" in fields:
            fields = ["hosts", "ports", "services", "tags", "labels"]

        self.fields = set(fields)

    def make(self, hosts: list[str], tables) -> Select:
        """
        tables must contain SQLAlchemy Table objects:
            tables.host
            tables.fingerprint
            tables.label
            tables.fingerprintlabel
            tables.hosttag
            tables.tag
        """

        host = tables["host"]
        fp = tables["fingerprint"]
        lbl = tables["label"]
        fpl = tables["fingerprintlabel"]
        ht = tables["hosttag"]
        tag = tables["tag"]

        hosts_cte = (
            select(host.c.ip, host.c.prefix, host.c.asn)
            .where(host.c.ip.in_(hosts))
            .cte("h")
        )

        stmt = select(hosts_cte.c.ip, hosts_cte.c.prefix, hosts_cte.c.asn)

        if "ports" in self.fields:
            ports_sub = (
                select(
                    fp.c.host.label("ip"),
                    func.group_concat(func.distinct(fp.c.port)).label("ports"),
                )
                .where(fp.c.host.in_(hosts))
                .group_by(fp.c.host)
                .subquery()
            )

            stmt = stmt.add_columns(ports_sub.c.ports)
            stmt = stmt.outerjoin(ports_sub, ports_sub.c.ip == hosts_cte.c.ip)

        if "services" in self.fields:
            labels_sub = (
                select(
                    fpl.c.fingerprint_id,
                    func.group_concat(lbl.c.name, ",").label("labels"),
                )
                .join(lbl, lbl.c.id == fpl.c.label_id)
                .group_by(fpl.c.fingerprint_id)
                .subquery()
            )

            fp_sub = (
                select(
                    fp.c.host.label("ip"),
                    fp.c.id,
                    fp.c.protocol,
                    fp.c.port,
                    fp.c.data,
                    labels_sub.c.labels,
                )
                .outerjoin(labels_sub, labels_sub.c.fingerprint_id == fp.c.id)
                .where(fp.c.host.in_(hosts))
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
                    ht.c.host.label("ip"),
                    func.group_concat(tag.c.name, ",").label("tags"),
                )
                .join(tag, tag.c.id == ht.c.tag_id)
                .where(ht.c.host.in_(hosts))
                .group_by(ht.c.host)
                .subquery()
            )

            stmt = stmt.add_columns(tags_sub.c.tags)
            stmt = stmt.outerjoin(tags_sub, tags_sub.c.ip == hosts_cte.c.ip)

        stmt = stmt.select_from(hosts_cte).order_by(hosts_cte.c.ip)

        return stmt


def new_info(fields: list[str]) -> InfoQueryBuilder:
    return InfoQueryBuilder(fields)
