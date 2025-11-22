import unittest
import json

from test import test_tools
from dice.helpers import new_label, new_fingerprint, new_fp_label

class TestRepository(unittest.TestCase):

    def test_add_records(self):
        repo = test_tools.load_test_repository()
        result = repo.get_records("2.2.2.2", "1.1.1.1")
        self.assertEqual(len(result.index), 2)

    def test_fingerprint_and_label(self):
        repo = test_tools.load_test_repository()
        targets = ("2.2.2.2", "1.1.1.1")

        # add the labels to the database
        lab = new_label("test", "test-label")
        repo.add_labels(lab)

        # get some records to fingerprint
        records = repo.get_records(*targets)

        # dummy fingerprint
        fps = []
        for _, record in records.iterrows():
            data = {"port": record["port"]}
            fp = new_fingerprint("test", record["ip"], record["id"], json.dumps(data))
            fps.append(fp)

        repo.fingerprint(*fps)

        # get the same fingerprints
        fingerprints = repo.get_fingerprints(*targets)

        # label fingerprints
        fp_labs = []
        for _, fp in fingerprints.iterrows():
            fp_labs.append(new_fp_label(fp["id"], lab.id))

        repo.label(*fp_labs)

        # evaluate
        summary = repo.summary()
        self.assertEqual(summary, {"fingerprinted": 2, "labelled": 2})