import duckdb

class Connector:
    def __init__(
        self,
        db_path: str | None,
        name: str = "-",
        read_only: bool = False,
        config: dict = {},
    ) -> None:
        self.name = name
        self.db_path: str = db_path if db_path else ":memory:"
        self.readonly = read_only
        self.config = config

        self.con: duckdb.DuckDBPyConnection | None = None

    def init_con(self, con, readonly: bool):
        self._extensions(con)
        self._pragmas(con)
        if not readonly:
            self._macros(con)

    def get_connection(self) -> duckdb.DuckDBPyConnection:
        if not self.con:
            self.con = self.new_connection()
        return self.con

    def new_connection(self) -> duckdb.DuckDBPyConnection:
        con = duckdb.connect(
            self.db_path, read_only=self.readonly
        )  # config=self.config)

        self.init_con(con, self.readonly)
        return con

    def with_connection(self, name: str) -> "Connector":
        return Connector(self.db_path, name, self.readonly, self.config)
    
    def attach(self, path: str, name: str) -> None:
        con = self.get_connection()
        con.execute(f"ATTACH '{path}' AS {name};")

    def copy(self, table: str, src: str) -> None:
        con = self.get_connection()
        con.execute(f"DROP TABLE IF EXISTS 'main.{table}'")
        con.execute(f"CREATE TABLE 'main.{table}' AS SELECT * FROM '{src}.{table}'")

    def _extensions(self, conn: duckdb.DuckDBPyConnection) -> None:
        conn.execute("INSTALL inet")
        conn.execute("LOAD inet")

    def _pragmas(self, conn: duckdb.DuckDBPyConnection) -> None:
        conn.execute("PRAGMA temp_directory='./duckdb_tmp';")
        conn.execute("PRAGMA threads=8;")
        conn.execute("PRAGMA memory_limit='10GB';")
        conn.execute("PRAGMA max_memory='10GB';")
        conn.execute("PRAGMA preserve_insertion_order=FALSE;")
        conn.execute("PRAGMA checkpoint_threshold='100GB';")  # very importante

    def _macros(self, conn: duckdb.DuckDBPyConnection) -> None:
        conn.execute("""
            CREATE OR REPLACE MACRO network_from_cidr(cidr_range) AS (
                cast(string_split(string_split(cidr_range, '/')[1], '.')[1] as bigint) * (256 * 256 * 256) +
                cast(string_split(string_split(cidr_range, '/')[1], '.')[2] as bigint) * (256 * 256      ) +
                cast(string_split(string_split(cidr_range, '/')[1], '.')[3] as bigint) * (256            ) +
                cast(string_split(string_split(cidr_range, '/')[1], '.')[4] as bigint)
            );
        """)

        conn.execute("""
            CREATE OR REPLACE MACRO broadcast_from_cidr(cidr_range) AS (
                cast(string_split(string_split(cidr_range, '/')[1], '.')[1] as bigint) * (256 * 256 * 256) +
                cast(string_split(string_split(cidr_range, '/')[1], '.')[2] as bigint) * (256 * 256      ) +
                cast(string_split(string_split(cidr_range, '/')[1], '.')[3] as bigint) * (256            ) +
                cast(string_split(string_split(cidr_range, '/')[1], '.')[4] as bigint)) + 
                cast(pow(256, (32 - cast(string_split(cidr_range, '/')[2] as bigint)) / 8) - 1 as bigint
            );
        """)

        conn.execute("""
            CREATE OR REPLACE MACRO ip_within_cidr(ip, cidr_range) AS (
                network_from_cidr(ip || '/32') >= network_from_cidr(cidr_range) AND network_from_cidr(ip || '/32') <= broadcast_from_cidr(cidr_range)
            );       
        """)

        conn.execute("""
            CREATE OR REPLACE FUNCTION inet_aton(ip) AS (
                cast(
                split_part(ip, '.', 1)::UINTEGER * 16777216 +
                split_part(ip, '.', 2)::UTINYINT * 65536 +
                split_part(ip, '.', 3)::UTINYINT * 256 +
                split_part(ip, '.', 4)::UTINYINT as UINTEGER)
            );
        """)

def new_connector(db: str, readonly: bool = False, name: str = "-") -> Connector:
    return Connector(db, read_only=readonly, name=name)
