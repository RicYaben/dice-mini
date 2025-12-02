from dataclasses import dataclass
from typing import Optional
from collections import OrderedDict

import ipaddress
from lark import Lark, Token, Transformer, Tree, v_args


@dataclass
class IPRange:
    start: int
    end: int


@dataclass
class Query:
    ip_range: IPRange
    filter: Optional["Expr"]  # top-level expression


class Expr:
    pass


@dataclass
class Comparison(Expr):
    field: str
    op: str
    value: str


@dataclass
class Not(Expr):
    expr: Expr


@dataclass
class BinaryOp(Expr):
    op: str  # "AND" or "OR"
    left: Expr
    right: Expr


def parse_ip_range(s: str) -> IPRange:
    s = s.strip()
    if "/" in s:
        net = ipaddress.ip_network(s, strict=False)
        return IPRange(int(net.network_address), int(net.broadcast_address))
    if "-" in s:
        a, b = s.split("-", 1)
        return IPRange(
            int(ipaddress.ip_address(a.strip())), int(ipaddress.ip_address(b.strip()))
        )
    ip = ipaddress.ip_address(s)
    return IPRange(int(ip), int(ip))


class QueryTransformer(Transformer):
    # Utility: fully unwrap lark Tree/Token to Python primitives
    def _unwrap(self, x):
        if isinstance(x, Tree):
            # Tree children should have already been transformed, so unwrap its one element
            if len(x.children) == 1:
                return self._unwrap(x.children[0])
            return [self._unwrap(c) for c in x.children]
        if isinstance(x, Token):
            return x.value
        return x

    # ---------- TOP LEVEL ----------
    def start(self, items):
        items = [self._unwrap(i) for i in items]

        ip = None
        filters = []

        if items and isinstance(items[0], str) and "/" in items[0]:
            ip = items[0]
            filters = items[1:]
        else:
            filters = items

        # Default IP
        if ip is None:
            ip = "0.0.0.0/0"

        # Normalize filters
        if len(filters) == 0:
            filters = {"type": "TRUE"}
        elif len(filters) == 1:
            filters = filters[0]
        else:
            filters = {"type": "AND", "children": filters}

        return {"ip": ip, "filters": filters}

    def ip(self, items):
        return str(items[0])

    # ---------- EXPRESSIONS ----------
    def or_expr(self, items):
        items = [self._unwrap(i) for i in items]
        if len(items) == 1:
            return items[0]
        return {"type": "OR", "children": items}

    def and_expr(self, items):
        items = [self._unwrap(i) for i in items]
        if len(items) == 1:
            return items[0]
        return {"type": "AND", "children": items}

    # ---------- TERMS & NEGATION ----------
    def negation(self, items):
        child = self._unwrap(items[0])
        return {"type": "NOT", "child": child}

    def atom(self, items):
        return self._unwrap(items[0])

    # ---------- CONDITIONS ----------
    @v_args(inline=True)
    def condition(self, field, op, value):
        return {
            "type": "COND",
            "field": str(field),
            "op": str(op),
            "value": self._value(value),
        }

    def _value(self, val):
        if isinstance(val, Token) and val.type == "STRING":
            return val.value.strip('"')
        return str(val)

    def value(self, items):
        return self._unwrap(items[0])


TABLE_MAP = {
    "port": "fingerprints",
    "service": "fingerprints",
    "tag": "tags",
    "label": "labels",
}

FIELD_MAP = {
    "tag": "name",
    "label": "name",
    "service": "protocol",
    "host": "ip",
    "port": "port",
}


class SQLBuilder:
    def __init__(self):
        self.joins = OrderedDict()

    def build(self, ast):
        "Returns a query for hosts"

        sql = ["SELECT DISTINCT(hosts.ip) FROM records_hosts AS hosts"]
        where = []

        # ---------- IP ----------
        ip = ast.get("ip", "0.0.0.0/0")  # default IP
        if "/" in ip:
            where.append(f"ip_within_cidr(hosts.ip, '{ip}')")
        else:
            where.append(f"hosts.ip = '{ip}'")

        # ---------- Filters ----------
        filters = ast.get("filters")
        if filters and filters["type"] != "TRUE":
            expr = self.visit(filters)
            if expr:
                where.append(expr)

        # ---------- JOINs ----------
        for join in self.joins:
            sql.append(join)

        # ---------- WHERE ----------
        if where:
            sql.append("WHERE " + " AND ".join(where))

        return "\n".join(sql)

    # ---------- Visitor ----------
    def visit(self, node):
        t = node["type"]

        if t == "AND":
            parts = [self.visit(c) for c in node["children"]]
            parts = [p for p in parts if p]
            return "(" + " AND ".join(parts) + ")"

        if t == "OR":
            parts = [self.visit(c) for c in node["children"]]
            parts = [p for p in parts if p]
            return "(" + " OR ".join(parts) + ")"

        if t == "NOT":
            return f"(NOT {self.visit(node['child'])})"

        if t == "COND":
            return self.condition(node)

        raise ValueError(f"Unknown node type: {t}")

    # ---------- Condition ----------
    def condition(self, node):
        field = node["field"]
        op = node["op"]
        value = node["value"]

        table_alias, column = self.resolve_field(field)
        self.ensure_join(table_alias)

        # Quote strings
        if isinstance(value, str):
            if not value.replace(".", "").isdigit():
                # treat as string
                value = value.strip('"')
                value = "'" + value.replace("'", "''") + "'"
            else:
                # numeric, keep as is
                pass

        # Map ":" to LIKE or =
        if op == ":":
            if isinstance(value, str) and value.startswith("'"):
                # string → LIKE
                return f"{table_alias}.{column} LIKE {value}"
            else:
                # numeric → =
                return f"{table_alias}.{column} = {value}"

        return f"{table_alias}.{column} {op} {value}"

    # ---------- Field / Table Resolution ----------
    def resolve_field(self, field):
        table = None
        for prefix, tbl in TABLE_MAP.items():
            if field == prefix or field.startswith(prefix + "."):
                table = tbl
                break
        if table is None:
            table = "hosts"

        if field in FIELD_MAP:
            column = FIELD_MAP[field]
        else:
            column = field.split(".")[-1]

        return table, column

    # ---------- JOINs ----------
    def ensure_join(self, table):
        if table == "hosts":
            return

        if table == "fingerprints":
            self.join("LEFT JOIN fingerprints ON fingerprints.host = hosts.ip")
        elif table == "tags":
            # join through host_tags
            self.join("LEFT JOIN host_tags ON host_tags.host = hosts.ip")
            self.join("LEFT JOIN tags ON tags.id = host_tags.tag_id")
        elif table == "labels":
            self.ensure_join("fingerprints")
            # join through host_labels
            self.join(
                "LEFT JOIN fingerprint_labels ON fingerprint_labels.fingerprint_id = fingerprints.id"
            )
            self.join("LEFT JOIN labels ON labels.id = fingerprint_labels.label_id")

    def join(self, j: str):
        self.joins[j] = None


def parse_query(query: str):
    if not query or not query.strip():
        raise ValueError("empty query")

    parser = Lark(get_grammar(), start="start", parser="lalr", lexer="contextual")
    tree = parser.parse(query)
    print(tree.pretty())
    ast = QueryTransformer().transform(tree)
    q = SQLBuilder().build(ast)
    return q


def get_grammar():
    return r"""
        %import common.CNAME
        %import common.INT
        %import common.ESCAPED_STRING
        %import common.WS
        %ignore WS

        IP      : /[0-9]{1,3}(\.[0-9]{1,3}){3}(\/[0-9]{1,2})?/
        FIELD   : /[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*/
        NUMBER  : INT
        STRING  : ESCAPED_STRING

        OP      : ">=" | "<=" | ">" | "<" | ":" | "="

        start: ip? query*

        ip: IP

        query: or_expr

        or_expr: and_expr ("or" and_expr)*
        and_expr: term+

        term: negation | atom

        negation: "-" atom      -> negation

        atom: condition
            | "(" or_expr ")"

        condition: FIELD OP value
        value: NUMBER | STRING | CNAME
    """


class Parser:
    def __init__(self, grammar: str, transformer: Transformer) -> None:
        self.parser = Lark(grammar, start="start", parser="lalr", lexer="contextual")
        self.transformer = transformer
        self.builder = SQLBuilder()

    def parse(self, q: str) -> str:
        tree = self.parser.parse(q)
        exp = self.transformer.transform(tree)
        return exp

    def to_sql(self, q: str):
        ast = self.parse(q)
        return self.builder.build(ast)


def new_transformer():
    return QueryTransformer()


def new_parser(grammar: str, transformer: Transformer) -> Parser:
    return Parser(grammar, transformer)


def make_parser() -> Parser:
    t = new_transformer()
    return Parser(get_grammar(), t)
