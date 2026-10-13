import csv
import hashlib
import os
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class DataIntegrityTests(unittest.TestCase):
    def test_manifest_defines_expected_files(self):
        with (ROOT / "data_manifests" / "dataset_manifest.csv").open(encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 16)
        self.assertEqual({int(row["file_id"]) for row in rows}, {97,98,99,100,105,106,107,108,118,119,120,121,130,131,132,133})
        self.assertEqual({int(row["load_hp"]) for row in rows}, {0,1,2,3})

    def test_local_data_hashes_when_data_dir_is_supplied(self):
        value = os.environ.get("AT172_DATA_DIR")
        if not value:
            self.skipTest("AT172_DATA_DIR not set; raw CWRU data are intentionally not distributed")
        data_dir = Path(value)
        with (ROOT / "data_manifests" / "dataset_manifest.csv").open(encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            path = data_dir / f"{row['file_id']}.mat"
            self.assertTrue(path.exists(), path)
            self.assertEqual(path.stat().st_size, int(row["bytes"]))
            self.assertEqual(sha256(path), row["sha256"])


if __name__ == "__main__":
    unittest.main()

