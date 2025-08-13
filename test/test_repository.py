import unittest
import tempfile
import os
import pickle

from dice.repo import load_repository, Repository
from dice.helpers import new_source, new_label, new_collection, new_fingerprint, new_fp_label

class TestRepository(unittest.TestCase):
    def load_repository(self) -> Repository :
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "results.jsonl")
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(
                    '{"ip":"2.2.2.2","port":4242,"data":{"test":{"status":"success","protocol":"test"}}}\r\n'+
                    '{"ip":"1.1.1.1","port":4242,"data":{"test":{"status":"success","protocol":"test"}}}\r\n'
                )
            
            # takes a name of the source and a path to a file
            src = new_source("zgrab2", fpath)
            return load_repository([src])

    def test_add_records(self):
        repo = self.load_repository()
        result = repo.get_records(["2.2.2.2", "1.1.1.1"])
        self.assertEqual(len(result.index), 2)

    def test_fingerprint_and_label(self):
        repo = self.load_repository()
        targets= ["2.2.2.2", "1.1.1.1"]

        # add the labels to the database
        lab = new_label("test", "test-label")
        repo.add_labels(lab)

        # get some records to fingerprint
        records = repo.get_records(targets)

        # dummy fingerprint
        fps = []
        for _, record in records.iterrows():
            data = {"port": record["port"]}
            fp = new_fingerprint("test", record["ip"], record["id"], pickle.dumps(data))
            fps.append(fp)

        repo.fingerprint(*fps)

        # get the same fingerprints
        fingerprints = repo.get_fingerprints(targets)

        # label fingerprints
        fp_labs = []
        for _, fp in fingerprints.iterrows():
            fp_labs.append(new_fp_label(fp["id"], lab.id))

        repo.label(*fp_labs)

        # evaluate
        summary = repo.summary()
        self.assertEqual(summary, {"fingerprinted": 2, "labelled": 2})


if __name__ == "__main__":
    unittest.main()