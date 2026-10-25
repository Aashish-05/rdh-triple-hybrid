"""Tests for deterministic content-based versioned split construction."""
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.create_clean_split_v2 import build_versioned_split


class TestCleanSplitV2(unittest.TestCase):
    def test_cross_partition_duplicates_are_excluded_and_within_train_remains(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            filenames_content = {
                "train-a.pgm": b"train-repeat",
                "train-b.pgm": b"train-repeat",
                "val.pgm": b"validation",
                "holdout-copy.pgm": b"train-repeat",
                "holdout-only.pgm": b"holdout",
                "test.pgm": b"test",
            }
            manifest = {}
            for filename, content in filenames_content.items():
                source = root / filename
                source.write_bytes(content)
                manifest[filename] = {"absolute_path": str(source)}
            splits = {
                "metadata": {"seed": 42},
                "train": ["train-a.pgm", "train-b.pgm"],
                "val": ["val.pgm"],
                "holdout": ["holdout-copy.pgm", "holdout-only.pgm"],
                "test": ["test.pgm"],
            }

            versioned, new_manifest = build_versioned_split(splits, manifest)

        self.assertEqual(versioned["splits"]["train"], ["train-a.pgm", "train-b.pgm"])
        self.assertEqual(versioned["splits"]["val"], ["val.pgm"])
        self.assertEqual(versioned["splits"]["holdout"], ["holdout-only.pgm"])
        self.assertEqual(versioned["splits"]["test"], ["test.pgm"])
        self.assertEqual(versioned["excluded_redundant"][0]["filename"], "holdout-copy.pgm")
        self.assertEqual(new_manifest["holdout-copy.pgm"]["split"], "excluded_redundant")
        self.assertEqual(new_manifest["train-a.pgm"]["sha256"], new_manifest["train-b.pgm"]["sha256"])
        self.assertEqual(versioned["metadata"]["counts"]["holdout"], 1)
