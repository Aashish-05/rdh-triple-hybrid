"""
Unit and Integration Tests for Image Loading, Grayscale Preprocessing, and EDA Integrity.
"""
import unittest
import sys
import json
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config
from src.data.loader import load_grayscale_image, RDHImageDataset


class TestLoaderAndEDA(unittest.TestCase):
    def setUp(self):
        self.cfg = get_config()
        with open(self.cfg.paths.manifest_path, "r") as f:
            self.manifest = json.load(f)
        with open(self.cfg.paths.splits_path, "r") as f:
            self.splits = json.load(f)

    def test_pgm_loading_preserves_raw_bytes(self):
        """Verify that loading an 8-bit PGM preserves raw pixel bytes without any distortion."""
        sample_name = self.splits["train"][0]
        sample_path = Path(self.manifest[sample_name]["absolute_path"])
        
        # Read raw binary payload directly from file
        with open(sample_path, "rb") as f:
            _ = f.readline()  # P5
            _ = f.readline()  # 512 512
            _ = f.readline()  # 255
            raw_bytes = f.read()

        raw_array = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((512, 512))
        
        # Load via loader function
        loaded_array, meta = load_grayscale_image(sample_path)
        
        self.assertEqual(meta["orig_mode"], "L")
        self.assertFalse(meta["converted_to_gray"])
        self.assertFalse(meta["resized"])
        self.assertEqual(loaded_array.shape, (512, 512))
        self.assertEqual(loaded_array.dtype, np.uint8)
        # Byte-for-byte exact equality
        np.testing.assert_array_equal(loaded_array, raw_array)

    def test_tiff_monochrome_loading(self):
        """Verify 1-channel TIFF loading preserves pixel values and dimensions."""
        boat_fn = "boat.512.tiff"
        self.assertIn(boat_fn, self.manifest)
        boat_path = Path(self.manifest[boat_fn]["absolute_path"])
        
        arr, meta = load_grayscale_image(boat_path)
        self.assertEqual(meta["orig_mode"], "L")
        self.assertFalse(meta["converted_to_gray"])
        self.assertFalse(meta["resized"])
        self.assertEqual(arr.shape, (512, 512))
        self.assertEqual(arr.dtype, np.uint8)
        self.assertGreaterEqual(arr.min(), 0)
        self.assertLessEqual(arr.max(), 255)

    def test_tiff_rgb_deterministic_conversion(self):
        """Verify RGB TIFF converts to grayscale deterministically via ITU-R BT.601."""
        baboon_fn = "4.2.03.tiff"
        self.assertIn(baboon_fn, self.manifest)
        baboon_path = Path(self.manifest[baboon_fn]["absolute_path"])
        
        arr, meta = load_grayscale_image(baboon_path)
        self.assertEqual(meta["orig_mode"], "RGB")
        self.assertTrue(meta["converted_to_gray"])
        self.assertFalse(meta["resized"])
        self.assertEqual(arr.shape, (512, 512))
        self.assertEqual(arr.dtype, np.uint8)

        # Independently verify against Pillow L mode convert
        with Image.open(baboon_path) as img:
            expected = np.array(img.convert("L"), dtype=np.uint8)
        np.testing.assert_array_equal(arr, expected)

    def test_no_unwanted_resizing(self):
        """Verify that 512x512 images are not resized, and mismatched dimensions raise ValueError."""
        sample_path = Path(self.manifest[self.splits["train"][0]]["absolute_path"])
        _, meta = load_grayscale_image(sample_path, target_size=(512, 512), allow_resize=False)
        self.assertFalse(meta["resized"])

        # When target_size is different and allow_resize=False, ValueError must be raised
        with self.assertRaises(ValueError):
            load_grayscale_image(sample_path, target_size=(256, 256), allow_resize=False)

    def test_rdh_dataset_loader_interface(self):
        """Verify PyTorch RDHImageDataset for train and test splits."""
        train_ds = RDHImageDataset(split="train", as_float=True, normalize_01=True)
        test_ds = RDHImageDataset(split="test", as_float=True, normalize_01=True)

        self.assertEqual(len(train_ds), 3500)
        self.assertEqual(len(test_ds), 19)

        # Check sample tensor output
        tensor, fn = train_ds[0]
        self.assertEqual(tensor.shape, (1, 512, 512))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertGreaterEqual(float(tensor.min()), 0.0)
        self.assertLessEqual(float(tensor.max()), 1.0)
        self.assertIn(fn, self.splits["train"])

    def test_eda_artifacts_exist_and_non_empty(self):
        """Verify that EDA plots and tables were generated properly."""
        figures_dir = self.cfg.paths.figures_dir
        tables_dir = self.cfg.paths.tables_dir

        for fig_name in ["eda_sample_grid.png", "eda_histograms.png", "eda_texture_complexity.png"]:
            p = figures_dir / fig_name
            self.assertTrue(p.exists(), f"Missing EDA figure: {fig_name}")
            self.assertGreater(p.stat().st_size, 50000, f"EDA figure too small/corrupted: {fig_name}")

        for tab_name in ["eda_dataset_summary.csv", "eda_image_statistics.csv", "eda_benchmark_complexity_ranking.csv", "eda_manifest.json"]:
            p = tables_dir / tab_name
            self.assertTrue(p.exists(), f"Missing EDA table: {tab_name}")
            self.assertGreater(p.stat().st_size, 100, f"EDA table is empty: {tab_name}")

    def test_source_files_unmodified(self):
        """Verify original source files on disk have identical SHA-256 hashes to manifest (no tampering)."""
        # Spot check first 5 train and first 5 test files
        check_files = self.splits["train"][:5] + self.splits["test"][:5]
        for fn in check_files:
            p = Path(self.manifest[fn]["absolute_path"])
            self.assertTrue(p.exists())
            sha = hashlib.sha256()
            with open(p, "rb") as f:
                while chunk := f.read(65536):
                    sha.update(chunk)
            self.assertEqual(sha.hexdigest(), self.manifest[fn]["sha256"], f"File was altered: {fn}")


if __name__ == "__main__":
    unittest.main()
