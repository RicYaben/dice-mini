import unittest

import dice
from dice import module
from test import test_tools

def test_cls_handler(mod: module.Module) -> None:
    fingerprints = mod.repo().get_fingerprints()
    for _, fp in fingerprints.iterrows():
        mod.label(fp["id"], "test-label")
    
def test_cls_init(mod: module.Module):
    mod.register_label("test-label")

def test_fignerprinter_module(mod: module.Module) -> None:
    records = mod.repo().get_records()
    for _, rec in records.iterrows():
        data = {"test": "test"}
        mod.fingerprint(rec, data, protocol="tets")

class TestEngine(unittest.TestCase):
    def test_engine(self):
        cmp_cls = dice.new_classifier(test_cls_handler, test_cls_init)
        cmp_fp = dice.new_fingerprinter(test_fignerprinter_module)

        engine = module.new_engine(cmp_fp, cmp_cls)
        srcs, clean = test_tools.make_test_sources()
        try:
            repo = engine.run(srcs)
            s = repo.summary()
            self.assertEqual(s, {"fingerprinted": 2, "labelled": 2})
        finally:
            clean()