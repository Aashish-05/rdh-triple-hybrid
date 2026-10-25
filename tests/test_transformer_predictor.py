"""
Unit tests for Window-Restricted Transformer Predictor Architecture and Pipeline.
"""
import unittest
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.predictors.transformer import (
    WindowTransformerPredictor,
    window_partition,
    window_unpartition,
)
from src.predictors.cnn import build_cnn_input
from src.data.partition import get_checkerboard_masks


class TestTransformerPredictor(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = WindowTransformerPredictor(
            in_channels=3,
            dim=32,
            window_size=8,
            num_heads=4,
            num_blocks=2,
            mlp_ratio=2.0,
        ).to(self.device)
        self.model.eval()

    def test_transformer_io_shapes(self):
        """Verify model handles (B, 3, H, W) where H, W are divisible by window_size."""
        B, H, W = 2, 64, 64
        x = torch.rand((B, 3, H, W), device=self.device)
        with torch.no_grad():
            out = self.model(x)
        self.assertEqual(out.shape, (B, 1, H, W))
        self.assertEqual(out.dtype, torch.float32)

    def test_window_partition_unpartition_invertibility(self):
        """Verify window_partition and window_unpartition are exact inverses."""
        x = torch.randn((2, 32, 64, 64), device=self.device)
        windows = window_partition(x, window_size=8)
        reconstructed = window_unpartition(windows, window_size=8, H=64, W=64, C=32)
        self.assertTrue(torch.equal(x, reconstructed))

    def test_target_pixel_isolation(self):
        """Verify target pixels are strictly excluded and cannot leak into input."""
        H, W = 32, 32
        cross_m, _ = get_checkerboard_masks(H, W)

        img1 = torch.full((1, 1, H, W), fill_value=100.0)
        img2 = img1.clone()
        target_idx = torch.from_numpy(cross_m)
        img2[0, 0, target_idx] = 250.0  # alter target pixels

        target_m_t = torch.from_numpy(cross_m).unsqueeze(0).unsqueeze(0)
        rp_prior = torch.full((1, 1, H, W), fill_value=0.4)

        x1 = build_cnn_input(img1, target_m_t, rp_prior)
        x2 = build_cnn_input(img2, target_m_t, rp_prior)

        # Context channels must be identical despite modified target pixels
        self.assertTrue(torch.all(x1[:, 0:1, target_idx] == 0.0))
        self.assertTrue(torch.all(x2[:, 0:1, target_idx] == 0.0))
        self.assertTrue(torch.equal(x1, x2), "Target pixel values leaked into Transformer input!")

    def test_output_range_and_determinism(self):
        """Verify model inference is strictly within [0.0, 255.0] and deterministic."""
        x = torch.rand((1, 3, 32, 32), device=self.device)
        with torch.no_grad():
            out1 = self.model(x)
            out2 = self.model(x)

        self.assertTrue(torch.equal(out1, out2), "Model inference is non-deterministic!")
        self.assertGreaterEqual(float(out1.min()), 0.0)
        self.assertLessEqual(float(out1.max()), 255.0)

    def test_checkpoint_save_and_load(self):
        """Verify checkpoint serialization and state restoration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test_tfp.pt"
            torch.save({"model_state_dict": self.model.state_dict()}, ckpt_path)

            loaded_model = WindowTransformerPredictor(
                in_channels=3,
                dim=32,
                window_size=8,
                num_heads=4,
                num_blocks=2,
            ).to(self.device)
            ckpt = torch.load(ckpt_path, weights_only=False)
            loaded_model.load_state_dict(ckpt["model_state_dict"])
            loaded_model.eval()

            x = torch.rand((1, 3, 32, 32), device=self.device)
            with torch.no_grad():
                out_orig = self.model(x)
                out_loaded = loaded_model(x)

            self.assertTrue(torch.equal(out_orig, out_loaded))

    def test_training_smoke_test(self):
        """Verify backward pass, finite loss, and parameter updates."""
        model = WindowTransformerPredictor(in_channels=3, dim=16, window_size=8, num_heads=2, num_blocks=1).to(self.device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        x = torch.rand((2, 3, 16, 16), device=self.device)
        y = torch.rand((2, 1, 16, 16), device=self.device) * 255.0
        mask = torch.ones((2, 1, 16, 16), device=self.device, dtype=torch.bool)

        initial_param = next(model.parameters()).clone()

        for _ in range(2):
            optimizer.zero_grad()
            pred = model(x)
            loss = torch.sum(torch.abs(pred - y) * mask.float()) / mask.sum()
            self.assertTrue(torch.isfinite(loss).item())
            loss.backward()
            optimizer.step()

        updated_param = next(model.parameters())
        self.assertFalse(torch.equal(initial_param, updated_param), "Parameters did not update!")


if __name__ == "__main__":
    unittest.main()
