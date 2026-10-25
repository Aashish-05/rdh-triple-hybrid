"""
Unit tests for CNN Predictor Architecture, Target Isolation, and Training Pipeline.
"""
import unittest
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.predictors.cnn import (
    CNNPixelPredictor,
    build_cnn_input,
    predict_cnn_image,
)
from src.data.partition import get_checkerboard_masks
from src.data.training_data import RDHCheckerboardDataset


class TestCNNPredictor(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNNPixelPredictor(in_channels=3, hidden_channels=32).to(self.device)
        self.model.eval()

    def test_cnn_io_shapes(self):
        """Verify model handles batched 3-channel input and returns (B, 1, H, W)."""
        B, H, W = 2, 64, 64
        x = torch.rand((B, 3, H, W), device=self.device)
        with torch.no_grad():
            out = self.model(x)
        self.assertEqual(out.shape, (B, 1, H, W))
        self.assertEqual(out.dtype, torch.float32)

    def test_target_pixel_isolation(self):
        """Verify target pixels are strictly zeroed out in the context channel."""
        H, W = 32, 32
        cross_m, dot_m = get_checkerboard_masks(H, W)
        
        # Test image 1
        img1 = torch.full((1, 1, H, W), fill_value=128.0)
        # Test image 2 (same context, different target pixels)
        img2 = img1.clone()
        target_indices = torch.from_numpy(cross_m)
        img2[0, 0, target_indices] = 255.0  # alter target pixel values

        target_m_t = torch.from_numpy(cross_m).unsqueeze(0).unsqueeze(0)
        rp_prior = torch.full((1, 1, H, W), fill_value=0.5)

        x1 = build_cnn_input(img1, target_m_t, rp_prior)
        x2 = build_cnn_input(img2, target_m_t, rp_prior)

        # Channel 0 at target locations must be strictly 0.0
        self.assertTrue(torch.all(x1[:, 0:1, target_indices] == 0.0))
        self.assertTrue(torch.all(x2[:, 0:1, target_indices] == 0.0))

        # Because context pixels were identical, x1 and x2 must be 100% identical!
        self.assertTrue(torch.equal(x1, x2), "Target pixel values leaked into CNN input!")

    def test_output_range_and_determinism(self):
        """Verify model outputs are within [0.0, 255.0] and deterministic."""
        x = torch.rand((1, 3, 32, 32), device=self.device)
        with torch.no_grad():
            out1 = self.model(x)
            out2 = self.model(x)

        self.assertTrue(torch.equal(out1, out2), "Model inference is non-deterministic!")
        self.assertGreaterEqual(float(out1.min()), 0.0)
        self.assertLessEqual(float(out1.max()), 255.0)

    def test_checkpoint_save_and_load(self):
        """Verify model state dict can be saved, reloaded, and produces identical output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test_model.pt"
            torch.save({"model_state_dict": self.model.state_dict()}, ckpt_path)

            loaded_model = CNNPixelPredictor(in_channels=3, hidden_channels=32).to(self.device)
            ckpt = torch.load(ckpt_path, weights_only=False)
            loaded_model.load_state_dict(ckpt["model_state_dict"])
            loaded_model.eval()

            x = torch.rand((1, 3, 32, 32), device=self.device)
            with torch.no_grad():
                out_orig = self.model(x)
                out_loaded = loaded_model(x)

            self.assertTrue(torch.equal(out_orig, out_loaded))

    def test_training_smoke_test(self):
        """Verify backpropagation, finite loss, and parameter updates on synthetic batch."""
        model = CNNPixelPredictor(in_channels=3, hidden_channels=16).to(self.device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        x = torch.rand((2, 3, 32, 32), device=self.device)
        y = torch.rand((2, 1, 32, 32), device=self.device) * 255.0
        mask = torch.ones((2, 1, 32, 32), device=self.device, dtype=torch.bool)

        initial_param = next(model.parameters()).clone()

        # Run 2 training steps
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
