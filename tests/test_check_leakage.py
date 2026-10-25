"""Unit tests for the all-split content-hash audit."""
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.check_leakage import find_content_duplicates


class TestAllSplitHashAudit(unittest.TestCase):
    def test_hashes_each_split_and_detects_intra_and_cross_split_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contents = {
                "train-a.pgm": b"duplicate",
                "train-b.pgm": b"duplicate",
                "val.pgm": b"validation",
                "holdout.pgm": b"duplicate",
                "test.pgm": b"test",
            }
            manifest = {}
            for filename, data in contents.items():
                path = root / filename
                path.write_bytes(data)
                manifest[filename] = {"absolute_path": str(path)}

            splits = {
                "train": ["train-a.pgm", "train-b.pgm"],
                "val": ["val.pgm"],
                "holdout": ["holdout.pgm"],
                "test": ["test.pgm"],
            }
            duplicates, missing = find_content_duplicates(splits, manifest)

        self.assertEqual(missing, [])
        self.assertEqual(len(duplicates), 1)
        self.assertEqual(
            duplicates[0]["files_by_split"],
            {"train": ["train-a.pgm", "train-b.pgm"], "holdout": ["holdout.pgm"]},
        )
