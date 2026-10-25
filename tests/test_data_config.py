"""
Unit tests for Configuration, Dataset Manifest, and Zero Data Leakage.
"""
import unittest
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config, ProjectConfig


class TestDataAndConfig(unittest.TestCase):
    def setUp(self):
        self.cfg = get_config()

    def test_config_loading(self):
        """Verify configuration initializes with required attributes and defaults."""
        self.assertIsInstance(self.cfg, ProjectConfig)
        self.assertEqual(self.cfg.dataset.train_size, 3500)
        self.assertEqual(self.cfg.dataset.val_size, 500)
        self.assertEqual(self.cfg.dataset.holdout_size, 6000)
        self.assertEqual(self.cfg.dataset.seed, 42)
        self.assertEqual(self.cfg.dataset.target_size, (512, 512))
        self.assertTrue(self.cfg.paths.bossbase_dir.exists())
        self.assertTrue(self.cfg.paths.uscsipi_dir.exists())

    def test_splits_file_structure(self):
        """Verify splits.json exists and contains exact partition counts."""
        splits_path = self.cfg.paths.splits_path
        self.assertTrue(splits_path.exists(), f"Splits file not found: {splits_path}")
        
        with open(splits_path, "r") as f:
            splits = json.load(f)
            
        self.assertIn("train", splits)
        self.assertIn("val", splits)
        self.assertIn("holdout", splits)
        self.assertIn("test", splits)
        
        self.assertEqual(len(splits["train"]), 3500)
        self.assertEqual(len(splits["val"]), 500)
        self.assertEqual(len(splits["holdout"]), 6000)
        self.assertEqual(len(splits["test"]), 19)

    def test_manifest_file_validity(self):
        """Verify dataset_manifest.json contains valid entries and correct dimensions."""
        manifest_path = self.cfg.paths.manifest_path
        self.assertTrue(manifest_path.exists(), f"Manifest file not found: {manifest_path}")
        
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
            
        self.assertGreaterEqual(len(manifest), 10019)
        
        # Spot-check train and test entries
        sample_train = next(e for e in manifest.values() if e["split"] == "train")
        self.assertEqual(sample_train["width"], 512)
        self.assertEqual(sample_train["height"], 512)
        self.assertIsNotNone(sample_train["sha256"])
        self.assertEqual(sample_train["mode"], "L")

        sample_test = next(e for e in manifest.values() if e["split"] == "test")
        self.assertEqual(sample_test["width"], 512)
        self.assertEqual(sample_test["height"], 512)
        self.assertIsNotNone(sample_test["sha256"])

    def test_train_val_test_separation(self):
        """Assert zero filename overlap and zero content hash overlap across splits."""
        with open(self.cfg.paths.splits_path, "r") as f:
            splits = json.load(f)

        with open(self.cfg.paths.manifest_path, "r") as f:
            manifest = json.load(f)

        train_files = set(splits["train"])
        val_files = set(splits["val"])
        test_files = set(splits["test"])
        holdout_files = set(splits["holdout"])

        # Filename disjointness
        self.assertEqual(len(train_files & val_files), 0, "Train and Val share filenames!")
        self.assertEqual(len(train_files & test_files), 0, "Train and Test share filenames!")
        self.assertEqual(len(val_files & test_files), 0, "Val and Test share filenames!")
        self.assertEqual(len(train_files & holdout_files), 0, "Train and Holdout share filenames!")
        self.assertEqual(len(val_files & holdout_files), 0, "Val and Holdout share filenames!")

        # Content SHA-256 hash disjointness
        train_hashes = {manifest[fn]["sha256"] for fn in splits["train"] if manifest[fn]["sha256"]}
        val_hashes = {manifest[fn]["sha256"] for fn in splits["val"] if manifest[fn]["sha256"]}
        test_hashes = {manifest[fn]["sha256"] for fn in splits["test"] if manifest[fn]["sha256"]}

        self.assertEqual(len(train_hashes & val_hashes), 0, "Train and Val share identical file content!")
        self.assertEqual(len(train_hashes & test_hashes), 0, "CRITICAL: Train and Test share identical content!")
        self.assertEqual(len(val_hashes & test_hashes), 0, "CRITICAL: Val and Test share identical content!")


if __name__ == "__main__":
    unittest.main()
