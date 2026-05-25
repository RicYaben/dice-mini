import unittest
import ujson

from dice.internal.modules import Module
from dice.internal.engine import new_engine
from dice.internal.components import new_classifier, new_fingerprinter
from dice.internal.constructors import new_label, new_fingerprint
from dice.sdk.query import query

from test.test_tools import load_test_repository, summary

def test_cls_handler(mod: Module) -> None:
    for fp in mod.repo().stream(query("fingerprint")):
        mod.store(new_label(fp["id"], "test-label"))
    
def test_cls_init(mod: Module):
    mod.register_label("test-label")

def test_fignerprinter_module(mod: Module) -> None:
    for rec in mod.repo().stream(query("zgrab2_records")):
        data = {"test": "test"}
        mod.store(new_fingerprint("test", rec["host"], rec["id"], rec["resource_id"], ujson.dumps(data), protocol="tets"))

class TestEngine(unittest.TestCase):
    def test_engine(self):
        
        cmp_cls = new_classifier(test_cls_handler, test_cls_init)
        cmp_fp = new_fingerprinter(test_fignerprinter_module)

        engine = new_engine(cmp_fp, cmp_cls)
        repo = load_test_repository()
        repo = engine.run(repo)

        s = summary(repo)
        self.assertEqual(s, {"fingerprinted": 2, "labelled": 2})