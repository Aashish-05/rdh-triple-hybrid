"""Tests for versioned split loading and frozen router provenance."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.split_manifest import (
    load_split_manifest,
    ordered_filename_sha256,
    validate_v2_thresholds,
)
from src.data.training_data import RDHCheckerboardDataset


class TestVersionedSplitManifest(unittest.TestCase):
    def test_loads_nested_v2_membership_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "train.pgm"
            Image.fromarray(np.full((512, 512), 73, dtype=np.uint8)).save(image_path)
            splits_path = root / "splits.json"
            manifest_path = root / "manifest.json"
            split_document = {
                "metadata": {"schema_version": 2},
                "splits": {
                    "train": ["train.pgm"],
                    "val": [],
                    "holdout": [],
                    "test": [],
                },
            }
            manifest = {
                "train.pgm": {
                    "absolute_path": str(image_path),
                    "split": "train",
                    "filename": "train.pgm",
                }
            }
            splits_path.write_text(json.dumps(split_document), encoding="utf-8")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            splits, loaded_manifest, loaded_document = load_split_manifest(
                splits_path, manifest_path
            )

            self.assertEqual(splits["train"], ["train.pgm"])
            self.assertEqual(loaded_manifest["train.pgm"]["absolute_path"], str(image_path))
            self.assertEqual(loaded_document["metadata"]["schema_version"], 2)

            dataset = RDHCheckerboardDataset(
                split="train",
                crop_size=32,
                seed=42,
                splits_path=splits_path,
                manifest_path=manifest_path,
            )
            sample = dataset[0]
            self.assertEqual(sample["filename"], "train.pgm")
            self.assertEqual(tuple(sample["input"].shape), (3, 32, 32))

    def test_accepts_thresholds_only_for_exact_validation_membership_and_models(self):
        validation_filenames = ["val-a.pgm", "val-b.pgm"]
        thresholds = {
            "dataset_version": "v2",
            "selection_split": "val",
            "validation_count": 2,
            "validation_filenames_sha256": ordered_filename_sha256(validation_filenames),
            "checkpoints": {
                "cnn_sha256": "cnn-hash",
                "transformer_sha256": "transformer-hash",
            },
        }

        validate_v2_thresholds(
            thresholds, validation_filenames, "cnn-hash", "transformer-hash"
        )

        changed_membership = ["val-b.pgm", "val-a.pgm"]
        with self.assertRaisesRegex(ValueError, "validation_filenames_sha256"):
            validate_v2_thresholds(
                thresholds, changed_membership, "cnn-hash", "transformer-hash"
            )
        with self.assertRaisesRegex(ValueError, "checkpoints.cnn_sha256"):
            validate_v2_thresholds(
                thresholds, validation_filenames, "different-cnn", "transformer-hash"
            )


if __name__ == "__main__":
    unittest.main()
