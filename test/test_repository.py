import unittest
import json
import pandas as pd

from dice.query import query_db
from test.test_tools import load_test_repository, summary
from dice.constructors import new_label, new_fingerprint, new_fp_label

class TestRepository(unittest.TestCase):

    def test_add_records(self):
        repo = load_test_repository()
        n = repo.query_count(query_db("host", ip_in= ["1.1.1.1", "2.2.2.2"]))
        self.assertEqual(n, 2)

    def test_fingerprint_and_label(self):
        repo = load_test_repository()
        targets = ("2.2.2.2", "1.1.1.1")

        # add the labels to the database
        lab = new_label("test", "test-label")
        repo.insert([lab])
        assert(lab.id != None)

        # dummy fingerprint
        fps = []
        for r in repo.stream(query_db("host", ip_in=targets)):
                data = {"port": r["port"]}
                fp = new_fingerprint("test", resource_id=r["resource_id"], record_id=r["id"], host=r["ip"], data=json.dumps(data))
                fps.append(fp)

        repo.insert(*fps)

        # label fingerprints
        fp_labs = []
        for _, fp in repo.stream(query_db("fingerprint", ip_in=targets)):
            fp_labs.append(new_fp_label(fp["id"], lab.id))

        repo.insert(*fp_labs)

        # evaluate
        summary = summary(repo)
        self.assertEqual(summary, {"fingerprinted": 2, "labelled": 2})