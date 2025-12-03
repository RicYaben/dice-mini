class InfoQueryBuilder:
    """
    Builds a single DuckDB SQL query to extract hosts, ports, services+labels, and tags
    in one pass using list_agg + struct_pack.
    """

    def __init__(self, fields: list[str]):
        self.fields = set(fields)
        if "all" in self.fields:
            self.fields = {"hosts", "ports", "services", "tags", "labels"}

    def make(self, hosts: list[str], db: str = "") -> str:
        host_list = ",".join(f"'{h}'" for h in hosts)
        db_prefix = f"{db}." if db else ""

        # ---------- Hosts fields ----------
        host_select = "h.ip, h.prefix, h.asn"

        # ---------- Ports ----------
        ports_select = ""
        ports_join = ""
        if "ports" in self.fields:
            ports_select = ", ports_sub.ports"
            ports_join = f"""
            LEFT JOIN (
                SELECT
                    f.host AS ip,
                    list(DISTINCT f.port ORDER BY f.port) AS ports
                FROM {db_prefix}fingerprints f
                WHERE f.host IN ({host_list})
                GROUP BY f.host
            ) AS ports_sub
            ON ports_sub.ip = h.ip
            """

        # ---------- Services + Labels ----------
        services_select = ""
        services_join = ""
        if "services" in self.fields:
            services_select = ", services_sub.services"
            services_join = f"""
            LEFT JOIN (
                SELECT
                    f.host AS ip,
                    list(
                        struct_pack(
                            protocol := f.protocol,
                            port := f.port,
                            data := f.data,
                            labels := COALESCE(lbl.labels, [])
                        )
                        ORDER BY f.protocol, f.port
                    ) AS services
                FROM (
                    SELECT *
                    FROM {db_prefix}fingerprints f
                    WHERE f.host IN ({host_list})
                ) AS f
                LEFT JOIN (
                    SELECT fl.fingerprint_id,
                        list(l.name ORDER BY l.name) AS labels
                    FROM {db_prefix}fingerprint_labels fl
                    JOIN {db_prefix}labels l ON l.id = fl.label_id
                    GROUP BY fl.fingerprint_id
                ) AS lbl
                ON lbl.fingerprint_id = f.id
                GROUP BY f.host
            ) AS services_sub
            ON services_sub.ip = h.ip
            """

        # ---------- Tags ----------
        tags_select = ""
        tags_join = ""
        if "tags" in self.fields:
            tags_select = ", tags_sub.tags"
            tags_join = f"""
            LEFT JOIN (
                SELECT
                    ht.host AS ip,
                    list(t.name ORDER BY t.name) AS tags
                FROM {db_prefix}host_tags ht
                JOIN {db_prefix}tags t ON t.id = ht.tag_id
                WHERE ht.host IN ({host_list})
                GROUP BY ht.host
            ) AS tags_sub
            ON tags_sub.ip = h.ip
            """

        # ---------- Final Query ----------
        sql = f"""
        SELECT
            {host_select}
            {ports_select}
            {services_select}
            {tags_select}
        FROM (
            SELECT DISTINCT ip, prefix, asn
            FROM {db_prefix}records_hosts AS hosts
            WHERE ip IN ({host_list})
        ) AS h
        {ports_join}
        {services_join}
        {tags_join}
        ORDER BY h.ip;
        """

        return sql.strip()


def new_info(fields: list[str]) -> InfoQueryBuilder:
    return InfoQueryBuilder(fields)
