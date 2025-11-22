import unittest
import ujson
from dice.ast import new_transformer, new_parser, get_grammar

class TestAST(unittest.TestCase):
    queries = [
        '0.0.0.0/0 port:80 port:22',
        'port>=1 port<=1000 (service:modbus or service:fox) -tag:anonymous',
        '192.168.1.1/32 label:"bad actor" -service:foo',
        #'1.2.3.4-1.2.3.6 port=50',
        '0.0.0.0/0 port:80 port>=1 (service:modbus or service:fox) -tag:anon',
        'port:50 service:iec104 -tag:anon'
    ]

    def test_grammar(self):
        t = new_transformer()
        parser = new_parser(get_grammar(), t)
        for e in self.queries:
            expr = parser.parse(e)
            print(ujson.dumps(expr, indent=4))

    def test_sql(self):
        t = new_transformer()
        parser = new_parser(get_grammar(), t)
        for e in self.queries:
            q = parser.to_sql(e)
            print(f'query: "{e}"')
            print(q+"\n")