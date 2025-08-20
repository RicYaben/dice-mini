import unittest
import pickle

from dice import repo, helpers, module
from test import test_tools

class TestClassifier:
    def __init__(self) -> None:
        self.lab = helpers.new_label("test", "test-label")
    def get_init(self) -> module.ModuleInit:
        def init(r: repo.Repository):
            r.add_labels(self.lab)
        return init
    
    def get_handler(self) -> module.ModuleHandler:
        def handler(r: repo.Repository) -> None:
            fingerprints = r.get_fingerprints()
            fp_labs = []
            for _, fp in fingerprints.iterrows():
                fp_labs.append(helpers.new_fp_label(fp["id"], self.lab.id))

                r.label(*fp_labs)
            return
        return handler

def test_fignerprinter_module(r: repo.Repository) -> None:
    records = r.get_records()
    fps = []
    for _, record in records.iterrows():
        port = record["port"]
        data = {"port": port}
        fp = helpers.new_fingerprint("test", record["ip"], record["id"], pickle.dumps(data), port=port)
        fps.append(fp)

    r.fingerprint(*fps)
    return

class TestEngine(unittest.TestCase):
    def test_engine(self):
        cl = TestClassifier()
        c_fact_cls = module.new_component_factory(module.M_CLASSIFIER, "cls-comp")
        cmp_cls = c_fact_cls.make_component(
            c_fact_cls.make_signature(
                "cls-sig", 
                c_fact_cls.make_module(
                    "cls-mod", 
                    cl.get_handler(),
                    cl.get_init()
                )
            )
        )

        c_fact_fp = module.new_component_factory(module.M_FINGERPRINTER, "fp-comp")
        cmp_fp = c_fact_fp.make_component(
            c_fact_fp.make_signature(
                "fp-sig", 
                c_fact_fp.make_module(
                    "fp-mod", 
                    test_fignerprinter_module
                )
            )
        )

        engine = module.new_engine(cmp_fp, cmp_cls)
        srcs, clean = test_tools.make_test_sources()
        try:
            repo = engine.run(srcs)
            s = repo.summary()

            # evaluate
            self.assertEqual(s, {"fingerprinted": 2, "labelled": 2})
        finally:
            clean()