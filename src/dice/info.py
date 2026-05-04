# TODO: this may need some love, the ORM changes a lot
# class InfoQueryBuilder:
#     """
#     Builds a single SQL query to extract hosts, ports, services+labels, and tags
#     in one pass using list_agg + struct_pack.
#     """

#     def __init__(self, fields: list[str]):
#         self.fields = set(fields)
#         if "all" in self.fields:
#             self.fields = {"hosts", "ports", "services", "tags", "labels"}

#     def make(self, hosts: list[str], db: str = "") -> str:
#         host_list = ",".join(f"'{h}'" for h in hosts)
#         db_prefix = f"{db}." if db else ""

#         # ---------- Hosts fields ----------
#         host_select = "h.ip, h.prefix, h.asn"

#         # ---------- Ports ----------
#         ports_select = ""
#         ports_join = ""
#         if "ports" in self.fields:
#             ports_select = ", ports_sub.ports"
#             ports_join = f"""
#             LEFT JOIN (
#                 SELECT
#                     f.host AS ip,
#                     list(DISTINCT f.port ORDER BY f.port) AS ports
#                 FROM {db_prefix}fingerprint f
#                 WHERE f.host IN ({host_list})
#                 GROUP BY f.host
#             ) AS ports_sub
#             ON ports_sub.ip = h.ip
#             """

#         # ---------- Services + Labels ----------
#         services_select = ""
#         services_join = ""
#         if "services" in self.fields:
#             services_select = ", services_sub.services"
#             services_join = f"""
#             LEFT JOIN (
#                 SELECT
#                     f.host AS ip,
#                     list(
#                         struct_pack(
#                             protocol := f.protocol,
#                             port := f.port,
#                             data := f.data,
#                             labels := COALESCE(lbl.labels, [])
#                         )
#                         ORDER BY f.protocol, f.port
#                     ) AS services
#                 FROM (
#                     SELECT *
#                     FROM {db_prefix}fingerprint f
#                     WHERE f.host IN ({host_list})
#                 ) AS f
#                 LEFT JOIN (
#                     SELECT fl.fingerprint_id,
#                         list(l.name ORDER BY l.name) AS labels
#                     FROM {db_prefix}fingerprintlabel fl
#                     JOIN {db_prefix}label l ON l.id = fl.label_id
#                     GROUP BY fl.fingerprint_id
#                 ) AS lbl
#                 ON lbl.fingerprint_id = f.id
#                 GROUP BY f.host
#             ) AS services_sub
#             ON services_sub.ip = h.ip
#             """

#         # ---------- Tags ----------
#         tags_select = ""
#         tags_join = ""
#         if "tags" in self.fields:
#             tags_select = ", tags_sub.tags"
#             tags_join = f"""
#             LEFT JOIN (
#                 SELECT
#                     ht.host AS ip,
#                     list(t.name ORDER BY t.name) AS tags
#                 FROM {db_prefix}hosttag ht
#                 JOIN {db_prefix}tag t ON t.id = ht.tag_id
#                 WHERE ht.host IN ({host_list})
#                 GROUP BY ht.host
#             ) AS tags_sub
#             ON tags_sub.ip = h.ip
#             """

#         # ---------- Final Query ----------
#         sql = f"""
#         SELECT
#             {host_select}
#             {ports_select}
#             {services_select}
#             {tags_select}
#         FROM (
#             SELECT DISTINCT ip, prefix, asn
#             FROM {db_prefix}host AS hosts
#             WHERE ip IN ({host_list})
#         ) AS h
#         {ports_join}
#         {services_join}
#         {tags_join}
#         ORDER BY h.ip;
#         """

#         return sql.strip()

from sqlalchemy import select, func, and_
from sqlalchemy.sql import Select


class InfoQueryBuilder:
    """
    Safe SQLAlchemy query builder (SQLite compatible)
    """

    def __init__(self, fields: list[str]):
        fields = list(dict.fromkeys(fields))  # preserve order

        if "all" in fields:
            fields = ["hosts", "ports", "services", "tags", "labels"]

        self.fields = set(fields)

    # -------------------------
    # MAIN QUERY
    # -------------------------
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

        # -------------------------
        # BASE HOST FILTER (safe IN clause)
        # -------------------------
        hosts_cte = (
            select(host.c.ip, host.c.prefix, host.c.asn)
            .where(host.c.ip.in_(hosts))
            .cte("h")
        )

        stmt = select(
            hosts_cte.c.ip,
            hosts_cte.c.prefix,
            hosts_cte.c.asn
        )

        # -------------------------
        # PORTS
        # -------------------------
        if "ports" in self.fields:
            ports_sub = (
                select(
                    fp.c.host.label("ip"),
                    func.group_concat(func.distinct(fp.c.port)).label("ports")
                )
                .where(fp.c.host.in_(hosts))
                .group_by(fp.c.host)
                .subquery()
            )

            stmt = stmt.add_columns(ports_sub.c.ports)
            stmt = stmt.outerjoin(ports_sub, ports_sub.c.ip == hosts_cte.c.ip)

        # -------------------------
        # SERVICES + LABELS
        # -------------------------
        if "services" in self.fields:
            labels_sub = (
                select(
                    fpl.c.fingerprint_id,
                    func.group_concat(lbl.c.name, ",").label("labels")
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
                    labels_sub.c.labels
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
                            "protocol", fp_sub.c.protocol,
                            "port", fp_sub.c.port,
                            "data", fp_sub.c.data,
                            "labels", func.coalesce(fp_sub.c.labels, "")
                        )
                    ).label("services")
                )
                .group_by(fp_sub.c.ip)
                .subquery()
            )

            stmt = stmt.add_columns(services_sub.c.services)
            stmt = stmt.outerjoin(services_sub, services_sub.c.ip == hosts_cte.c.ip)

        # -------------------------
        # TAGS
        # -------------------------
        if "tags" in self.fields:
            tags_sub = (
                select(
                    ht.c.host.label("ip"),
                    func.group_concat(tag.c.name, ",").label("tags")
                )
                .join(tag, tag.c.id == ht.c.tag_id)
                .where(ht.c.host.in_(hosts))
                .group_by(ht.c.host)
                .subquery()
            )

            stmt = stmt.add_columns(tags_sub.c.tags)
            stmt = stmt.outerjoin(tags_sub, tags_sub.c.ip == hosts_cte.c.ip)

        # -------------------------
        # FINAL
        # -------------------------
        stmt = stmt.select_from(hosts_cte).order_by(hosts_cte.c.ip)

        return stmt


def new_info(fields: list[str]) -> InfoQueryBuilder:
    return InfoQueryBuilder(fields)
