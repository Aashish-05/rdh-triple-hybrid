"""
Unit tests for Checkerboard Partitioning and Rhombus Predictor (RP).
"""
import unittest
import sys
from pathlib import Path
import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.partition import (
    get_checkerboard_masks,
    split_image_cross_dot,
    reconstruct_image_from_cross_dot,
)
from src.predictors.rhombus import (
    predict_rhombus_single,
    predict_rhombus,
    compute_prediction_errors,
)


class TestCheckerboardPartition(unittest.TestCase):
    def test_masks_disjoint_and_exhaustive(self):
        """Test that Cross and Dot masks partition the image space with zero overlap and full coverage."""
        shapes_to_test = [(512, 512), (5, 5), (4, 7), (1, 1), (2, 2), (1, 10)]
        for H, W in shapes_to_test:
            cm, dm = get_checkerboard_masks(H, W, as_tensor=False)
            
            # Disjointness
            self.assertFalse(np.any(cm & dm), f"Overlap found in shape ({H}, {W})")
            # Exhaustiveness
            self.assertTrue(np.all(cm | dm), f"Uncovered pixel in shape ({H}, {W})")
            # Pixel count conservation
            self.assertEqual(int(cm.sum() + dm.sum()), H * W)

            # PyTorch tensor variant
            t_cm, t_dm = get_checkerboard_masks(H, W, as_tensor=True)
            self.assertFalse(torch.any(t_cm & t_dm).item())
            self.assertTrue(torch.all(t_cm | t_dm).item())

    def test_exact_reconstruction(self):
        """Verify that splitting and reconstructing returns byte-for-byte identical images."""
        rng = np.random.RandomState(42)
        test_shapes = [(512, 512), (7, 9), (1, 1), (10, 10)]

        for H, W in test_shapes:
            # NumPy test
            orig = rng.randint(0, 256, size=(H, W), dtype=np.uint8)
            cross_vals, dot_vals, _ = split_image_cross_dot(orig)
            recon = reconstruct_image_from_cross_dot(cross_vals, dot_vals, H, W, dtype=np.uint8)
            np.testing.assert_array_equal(orig, recon)

            # PyTorch test
            t_orig = torch.from_numpy(orig)
            t_cross, t_dot, _ = split_image_cross_dot(t_orig)
            t_recon = reconstruct_image_from_cross_dot(t_cross, t_dot, H, W)
            self.assertTrue(torch.equal(t_orig, t_recon))


class TestRhombusPredictor(unittest.TestCase):
    def test_rp_hand_calculated_toy_example(self):
        """Verify RP formula on a 3x3 hand-calculated grid."""
        # Grid:
        # [ 10,  20,  30 ]
        # [ 40,  99,  60 ]   <- center pixel (1, 1) is 99 (Cross set: (1+1)%2 == 0)
        # [ 70,  80,  90 ]
        toy = np.array([
            [10, 20, 30],
            [40, 99, 60],
            [70, 80, 90],
        ], dtype=np.uint8)

        # Orthogonal neighbors of (1, 1): Top=20, Bottom=80, Left=40, Right=60
        # Expected average = (20 + 80 + 40 + 60) / 4 = 200 / 4 = 50
        # Formula: (200 + 2) // 4 = 50
        pred_img, eval_mask = predict_rhombus(toy, target_set="cross", exclude_border=True)
        
        self.assertTrue(eval_mask[1, 1])
        self.assertEqual(pred_img[1, 1], 50)

        # Prediction error: 99 - 50 = +49
        errors = compute_prediction_errors(toy, pred_img, eval_mask)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0], 49)

    def test_rp_half_up_tie_breaking(self):
        """Verify exact half-up integer rounding behavior."""
        # 1. Sum = 45, avg = 11.25 -> 11
        pred1 = predict_rhombus_single([10, 11, 12, 12])
        self.assertEqual(pred1, 11)

        # 2. Sum = 46, avg = 11.5 -> 12 (half-up)
        pred2 = predict_rhombus_single([10, 11, 12, 13])
        self.assertEqual(pred2, 12)

        # 3. Sum = 47, avg = 11.75 -> 12
        pred3 = predict_rhombus_single([10, 11, 13, 13])
        self.assertEqual(pred3, 12)

    def test_target_pixel_isolation(self):
        """Prove that the target pixel is NEVER used in its own prediction (zero leakage)."""
        base_grid = np.array([
            [20, 30, 40],
            [50,  0, 70],
            [80, 90, 100],
        ], dtype=np.uint8)

        # Test varying target pixel values at (1, 1)
        test_values = [0, 50, 128, 200, 255]
        predictions = []
        for val in test_values:
            grid = base_grid.copy()
            grid[1, 1] = val
            pred_img, _ = predict_rhombus(grid, target_set="cross", exclude_border=True)
            predictions.append(int(pred_img[1, 1]))

        # Expected: (30 + 90 + 50 + 70 + 2) // 4 = 242 // 4 = 60
        for p in predictions:
            self.assertEqual(p, 60, "Target pixel leaked into prediction!")

    def test_border_handling_options(self):
        """Verify border exclusion vs fallback border prediction."""
        toy = np.array([
            [10, 20, 30],
            [40, 50, 60],
            [70, 80, 90],
        ], dtype=np.uint8)

        # With exclude_border=True, only interior (1, 1) is evaluated
        _, eval_mask_ex = predict_rhombus(toy, target_set="both", exclude_border=True)
        self.assertEqual(eval_mask_ex.sum(), 1)
        self.assertTrue(eval_mask_ex[1, 1])

        # With exclude_border=False, all 9 pixels are evaluated
        pred_all, eval_mask_all = predict_rhombus(toy, target_set="both", exclude_border=False)
        self.assertEqual(eval_mask_all.sum(), 9)

        # Corner (0, 0) has 2 neighbors: right=20, bottom=40 -> (60 + 1) // 2 = 30
        self.assertEqual(pred_all[0, 0], 30)
        # Edge (0, 1) has 3 neighbors: left=10, right=30, bottom=50 -> (90 + 1) // 3 = 30
        self.assertEqual(pred_all[0, 1], 30)

    def test_prediction_bounds_and_determinism(self):
        """Verify output is in [0, 255] uint8 and deterministic."""
        rng = np.random.RandomState(123)
        img = rng.randint(0, 256, size=(64, 64), dtype=np.uint8)

        pred1, mask1 = predict_rhombus(img, target_set="both", exclude_border=False)
        pred2, mask2 = predict_rhombus(img, target_set="both", exclude_border=False)

        # Determinism
        np.testing.assert_array_equal(pred1, pred2)
        np.testing.assert_array_equal(mask1, mask2)

        # Bounds
        self.assertEqual(pred1.dtype, np.uint8)
        self.assertGreaterEqual(int(pred1.min()), 0)
        self.assertLessEqual(int(pred1.max()), 255)


if __name__ == "__main__":
    unittest.main()
