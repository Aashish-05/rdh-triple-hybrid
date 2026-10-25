"""
Unit Tests for Hybrid Complexity Router.

Tests cover:
  1. Boundary threshold routing (C=T1 -> CNN, not RP; C=T2 -> TFP, not CNN)
  2. Deterministic routing (same image -> same routing map always)
  3. Exhaustive routing codes (evaluated pixels: only 0/1/2; border: -1)
  4. Zero target leakage (complexity unchanged when target pixels change)
  5. Dual-hybrid mode (T2=1e9 -> no TFP assignments, only RP/CNN)
  6. RP-only mode (T1=1e9 -> all RP, code=0)
  7. Eval mask consistency with RP predictor mask
  8. router_thresholds.json schema validation (if file exists)
"""
import sys
import json
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.predictors.router import (
    compute_local_rhombus_complexity,
    compute_routing_map,
)
from src.predictors.rhombus import predict_rhombus
from src.data.partition import get_checkerboard_masks


class TestRouterThresholdBoundaries(unittest.TestCase):
    """Test 1 — Boundary values for T1 and T2."""

    def _make_uniform_complexity_image(self, H: int, W: int, value: float) -> np.ndarray:
        """
        Construct a synthetic image whose interior complexity is approximately `value`.
        Complexity = 0.5 * (gradient + std).
        Setting alternating rows of 0 and v gives a gradient of v/2 and std ~= v/2,
        so complexity ~= v/2. We solve by iteration: use a simple 0/v pattern.
        For exact control, we choose a constant-gradient pattern:
          row i = i * step % 256  so left-right neighbors differ by `step`.
        """
        arr = np.zeros((H, W), dtype=np.uint8)
        # Fill so that top-bottom difference is approximately 2*value, left-right ~=0
        # C = 0.5 * (|top-bottom| + std4) / 2
        # Simpler: just set specific pixels in a 5x5 test.
        return arr

    def test_complexity_at_T1_routes_to_cnn(self):
        """
        A pixel with complexity exactly == T1 must route to CNN (code 1), not RP (code 0).
        (Routing: C < T1 -> RP; C >= T1 -> CNN/TFP)
        """
        H, W = 8, 8
        # Build a 4-neighbor context that produces known complexity.
        # complexity = 0.5 * (grad + std4)
        # Neighbors: top=0, bottom=10, left=5, right=5
        #   grad = 0.5*(|0-10| + |5-5|) = 5.0
        #   std4 = std([0,10,5,5]) = std([0,10,5,5])
        std4 = float(np.std([0, 10, 5, 5]))
        expected_c = 0.5 * (5.0 + std4)

        img = np.zeros((H, W), dtype=np.uint8)
        # Set specific interior pixel (3,3) with known neighbors:
        #  top=(2,3), bottom=(4,3), left=(3,2), right=(3,4)
        img[2, 3] = 0    # top
        img[4, 3] = 10   # bottom
        img[3, 2] = 5    # left
        img[3, 4] = 5    # right
        img[3, 3] = 99   # target pixel (must NOT affect complexity)

        c_map = compute_local_rhombus_complexity(img, exclude_border=True)
        c_val = c_map[3, 3]

        self.assertAlmostEqual(c_val, expected_c, places=6,
                               msg=f"Complexity at (3,3) should be {expected_c:.4f}, got {c_val:.4f}")

        T1 = expected_c
        T2 = expected_c + 50.0

        # Build a fake eval_mask: only pixel (3,3) is evaluated
        eval_mask = np.zeros((H, W), dtype=bool)
        eval_mask[3, 3] = True

        routing = compute_routing_map(c_map, T1, T2, eval_mask)

        self.assertEqual(
            int(routing[3, 3]), 1,
            f"C=={T1:.4f}==T1 should route to CNN (code 1), got {routing[3,3]}"
        )

    def test_complexity_at_T2_routes_to_tfp(self):
        """
        A pixel with complexity exactly == T2 must route to TFP (code 2), not CNN (code 1).
        (Routing: C >= T2 -> TFP)
        """
        H, W = 8, 8
        img = np.zeros((H, W), dtype=np.uint8)
        img[2, 3] = 0
        img[4, 3] = 10
        img[3, 2] = 5
        img[3, 4] = 5

        c_map = compute_local_rhombus_complexity(img, exclude_border=True)
        c_val = c_map[3, 3]

        T1 = c_val - 20.0
        T2 = c_val  # complexity exactly equals T2

        eval_mask = np.zeros((H, W), dtype=bool)
        eval_mask[3, 3] = True

        routing = compute_routing_map(c_map, T1, T2, eval_mask)

        self.assertEqual(
            int(routing[3, 3]), 2,
            f"C=={T2:.4f}==T2 should route to TFP (code 2), got {routing[3,3]}"
        )


class TestRouterDeterminism(unittest.TestCase):
    """Test 2 — Routing map is deterministic for the same input."""

    def test_deterministic_routing(self):
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
        _, eval_mask = predict_rhombus(img, target_set="both", exclude_border=True)
        c_map = compute_local_rhombus_complexity(img, exclude_border=True)

        r1 = compute_routing_map(c_map, T1=3.0, T2=12.0, eval_mask=eval_mask)
        r2 = compute_routing_map(c_map, T1=3.0, T2=12.0, eval_mask=eval_mask)

        np.testing.assert_array_equal(r1, r2, err_msg="Routing map is non-deterministic!")


class TestRouterExhaustiveCodes(unittest.TestCase):
    """Test 3 — Evaluated pixels: only 0/1/2. Border pixels: -1."""

    def test_predictor_codes_exhaustive(self):
        rng = np.random.default_rng(1)
        img = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
        _, eval_mask = predict_rhombus(img, target_set="both", exclude_border=True)
        c_map = compute_local_rhombus_complexity(img, exclude_border=True)

        routing = compute_routing_map(c_map, T1=3.0, T2=12.0, eval_mask=eval_mask)

        valid_codes = {0, 1, 2}
        interior_codes = set(routing[eval_mask].tolist())
        self.assertTrue(
            interior_codes.issubset(valid_codes),
            f"Evaluated pixels contain invalid codes: {interior_codes - valid_codes}"
        )

        border_codes = set(routing[~eval_mask].tolist())
        self.assertEqual(
            border_codes, {-1},
            f"Non-evaluated pixels should all be -1, found: {border_codes}"
        )

    def test_routing_covers_all_evaluated_pixels(self):
        """Every pixel in eval_mask must receive exactly one of {0, 1, 2}."""
        rng = np.random.default_rng(2)
        img = rng.integers(0, 256, size=(32, 32), dtype=np.uint8)
        _, eval_mask = predict_rhombus(img, target_set="both", exclude_border=True)
        c_map = compute_local_rhombus_complexity(img, exclude_border=True)
        routing = compute_routing_map(c_map, T1=2.0, T2=8.0, eval_mask=eval_mask)

        evaluated_routing_codes = routing[eval_mask]
        self.assertTrue(
            np.all(evaluated_routing_codes >= 0),
            "Some evaluated pixels received code -1 (should be 0, 1, or 2)"
        )
        assigned_count = int((routing == 0).sum() + (routing == 1).sum() + (routing == 2).sum())
        self.assertEqual(
            assigned_count, int(eval_mask.sum()),
            "Total assigned pixels != total evaluated pixels"
        )


class TestRouterZeroTargetLeakage(unittest.TestCase):
    """Test 4 — Complexity map must not depend on target pixel values."""

    def test_complexity_excludes_target_pixel(self):
        """
        Change the target pixels; verify complexity map does not change.
        Complexity uses only the 4 orthogonal neighbors (context set),
        not the target pixel itself.
        """
        rng = np.random.default_rng(3)
        img_base = rng.integers(0, 256, size=(32, 32), dtype=np.uint8)

        # Identify target (Cross) pixels
        cross_mask, dot_mask = get_checkerboard_masks(32, 32)

        # Create a modified image with different target pixel values
        img_modified = img_base.copy()
        img_modified[cross_mask] = 255 - img_base[cross_mask]  # invert all cross pixels

        c_map_base = compute_local_rhombus_complexity(img_base, exclude_border=True)
        c_map_mod = compute_local_rhombus_complexity(img_modified, exclude_border=True)

        # Complexity at CROSS pixel locations: context is always DOT pixels.
        # Since we changed CROSS pixels only, and complexity uses the 4 orthogonal neighbors
        # (which are all DOT pixels), the complexity map SHOULD differ for cross pixels
        # because changing cross pixels changes the NEIGHBORS of dot pixels.
        #
        # The correct leakage test is: change a pixel X; its OWN complexity should not
        # depend on itself. The complexity_map[i,j] depends on img[i±1,j] and img[i,j±1]
        # NOT on img[i,j]. So changing img[i,j] should NOT affect complexity_map[i,j].
        #
        # We test this by modifying only the target pixel at position (4,4) (if cross):
        img_single = img_base.copy()
        r, c = 4, 4
        img_single[r, c] = 255 - img_base[r, c]

        c_map_single_base = compute_local_rhombus_complexity(img_base, exclude_border=True)
        c_map_single_mod = compute_local_rhombus_complexity(img_single, exclude_border=True)

        # The complexity at (r, c) should NOT change (it doesn't use img[r,c])
        self.assertAlmostEqual(
            c_map_single_base[r, c], c_map_single_mod[r, c], places=10,
            msg=f"Complexity at ({r},{c}) changed when pixel ({r},{c}) was modified — target leakage!"
        )


class TestRouterDualHybridMode(unittest.TestCase):
    """Test 5 — Dual hybrid mode: T2=1e9 means no TFP assignments."""

    def test_dual_hybrid_mode_no_tfp(self):
        rng = np.random.default_rng(4)
        img = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
        _, eval_mask = predict_rhombus(img, target_set="both", exclude_border=True)
        c_map = compute_local_rhombus_complexity(img, exclude_border=True)

        routing = compute_routing_map(c_map, T1=3.0, T2=1e9, eval_mask=eval_mask)

        tfp_assigned = int((routing == 2).sum())
        self.assertEqual(
            tfp_assigned, 0,
            f"T2=1e9 should assign zero pixels to TFP, but got {tfp_assigned}"
        )

        interior_codes = set(routing[eval_mask].tolist())
        self.assertTrue(
            interior_codes.issubset({0, 1}),
            f"Dual hybrid should only use codes 0 (RP) and 1 (CNN), found: {interior_codes}"
        )


class TestRouterRPOnlyMode(unittest.TestCase):
    """Test 6 — RP-only mode: T1=1e9 means all evaluated pixels route to RP (code=0)."""

    def test_rp_only_mode(self):
        rng = np.random.default_rng(5)
        img = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
        _, eval_mask = predict_rhombus(img, target_set="both", exclude_border=True)
        c_map = compute_local_rhombus_complexity(img, exclude_border=True)

        routing = compute_routing_map(c_map, T1=1e9, T2=2e9, eval_mask=eval_mask)

        rp_count = int((routing == 0).sum())
        total_eval = int(eval_mask.sum())
        self.assertEqual(
            rp_count, total_eval,
            f"T1=1e9 should assign all {total_eval} evaluated pixels to RP (code=0), "
            f"but only {rp_count} were assigned RP"
        )


class TestRouterEvalMaskConsistency(unittest.TestCase):
    """Test 7 — Routing eval_mask matches RP predictor eval_mask exactly."""

    def test_eval_masks_match_between_rp_and_routing(self):
        """
        The eval_mask from predict_rhombus (exclude_border=True) must be used
        for routing. This test verifies they have the same shape and coverage.
        """
        for H, W in [(32, 32), (64, 64), (512, 512)]:
            with self.subTest(H=H, W=W):
                img = np.zeros((H, W), dtype=np.uint8)
                _, eval_mask = predict_rhombus(img, target_set="both", exclude_border=True)
                c_map = compute_local_rhombus_complexity(img, exclude_border=True)
                routing = compute_routing_map(c_map, T1=3.0, T2=12.0, eval_mask=eval_mask)

                # Border (row/col 0 and H-1/W-1) must all be -1
                self.assertTrue(
                    np.all(routing[0, :] == -1),
                    f"Top border row must be -1 for {H}x{W}"
                )
                self.assertTrue(
                    np.all(routing[-1, :] == -1),
                    f"Bottom border row must be -1 for {H}x{W}"
                )
                self.assertTrue(
                    np.all(routing[:, 0] == -1),
                    f"Left border column must be -1 for {H}x{W}"
                )
                self.assertTrue(
                    np.all(routing[:, -1] == -1),
                    f"Right border column must be -1 for {H}x{W}"
                )

                # Interior must have only valid codes
                interior = routing[1:-1, 1:-1]
                valid = np.all((interior >= 0) & (interior <= 2))
                self.assertTrue(valid, f"Interior of routing map has invalid codes for {H}x{W}")


class TestRouterThresholdsJSON(unittest.TestCase):
    """Test 8 — Tuned thresholds JSON has required keys (if file exists)."""

    def test_tuned_thresholds_json_schema(self):
        """
        If router_thresholds.json exists (i.e. tune_router.py has been run),
        verify it has all required keys and valid T1/T2 values.
        """
        tables_dir = PROJECT_ROOT / "outputs" / "tables"
        json_path = tables_dir / "router_thresholds.json"

        if not json_path.exists():
            self.skipTest(
                "router_thresholds.json not found — run scripts/tune_router.py first. Skipping."
            )

        with open(json_path, "r") as f:
            data = json.load(f)

        required_keys = ["T1", "T2", "val_mae", "objective", "T1_candidates", "T2_candidates"]
        for key in required_keys:
            self.assertIn(key, data, f"Missing required key '{key}' in router_thresholds.json")

        T1 = data["T1"]
        T2_raw = data["T2"]
        T2 = float("inf") if T2_raw == "inf" else float(T2_raw)

        self.assertGreater(T1, 0.0, "T1 must be positive")
        self.assertGreater(T2, T1, f"T2 ({T2}) must be > T1 ({T1})")
        self.assertGreater(data["val_mae"], 0.0, "val_mae must be positive")

        val_mae = data["val_mae"]
        self.assertLess(val_mae, 20.0, f"val_mae={val_mae} is suspiciously large — check tuning")

        print(f"\n  [✓] router_thresholds.json: T1={T1}, T2={T2_raw}, val_mae={val_mae:.4f}")


class TestComplexityMapProperties(unittest.TestCase):
    """Additional sanity tests for the complexity map itself."""

    def test_uniform_image_has_zero_complexity(self):
        """A uniform (constant) image has zero gradient and zero std -> complexity = 0."""
        img = np.full((32, 32), fill_value=128, dtype=np.uint8)
        c_map = compute_local_rhombus_complexity(img, exclude_border=True)
        interior = c_map[1:-1, 1:-1]
        np.testing.assert_array_almost_equal(
            interior, 0.0, decimal=10,
            err_msg="Uniform image should have zero complexity everywhere"
        )

    def test_complexity_map_shape(self):
        """Complexity map must have the same shape as the input image."""
        for H, W in [(16, 16), (32, 64), (512, 512)]:
            img = np.zeros((H, W), dtype=np.uint8)
            c_map = compute_local_rhombus_complexity(img)
            self.assertEqual(c_map.shape, (H, W))

    def test_complexity_non_negative(self):
        """Complexity values must always be >= 0."""
        rng = np.random.default_rng(99)
        img = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
        c_map = compute_local_rhombus_complexity(img, exclude_border=True)
        self.assertTrue(
            np.all(c_map >= 0.0),
            f"Complexity map has negative values (min={c_map.min():.4f})"
        )

    def test_small_image_no_crash(self):
        """Images with H<=2 or W<=2 should return all-zero complexity without crashing."""
        for H, W in [(1, 1), (2, 2), (2, 64), (64, 2)]:
            img = np.zeros((H, W), dtype=np.uint8)
            c_map = compute_local_rhombus_complexity(img, exclude_border=True)
            self.assertEqual(c_map.shape, (H, W))
            np.testing.assert_array_equal(c_map, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
