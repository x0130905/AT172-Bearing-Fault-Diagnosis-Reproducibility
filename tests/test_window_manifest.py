import csv
import unittest
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class WindowManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT / "data_manifests" / "domain_split_manifest.csv").open(encoding="utf-8-sig") as stream:
            cls.rows = list(csv.DictReader(stream))

    def test_counts_and_boundaries(self):
        self.assertEqual(len(self.rows), 1280)
        counts = Counter(row["split"] for row in self.rows)
        self.assertEqual(counts, {"train": 640, "validation": 320, "unseen_test": 320})
        by_file = Counter(int(row["file_id"]) for row in self.rows)
        self.assertTrue(all(value == 80 for value in by_file.values()))
        self.assertTrue(all(int(row["end_exclusive"]) - int(row["start"]) == 1024 for row in self.rows))

    def test_non_overlapping_windows(self):
        grouped = defaultdict(list)
        for row in self.rows:
            grouped[int(row["file_id"])].append((int(row["start"]), int(row["end_exclusive"])))
        for intervals in grouped.values():
            intervals.sort()
            for left, right in zip(intervals[:-1], intervals[1:]):
                self.assertLessEqual(left[1], right[0])

    def test_files_do_not_cross_splits(self):
        file_splits = defaultdict(set)
        for row in self.rows:
            file_splits[int(row["file_id"])].add(row["split"])
        self.assertTrue(all(len(value) == 1 for value in file_splits.values()))


if __name__ == "__main__":
    unittest.main()

