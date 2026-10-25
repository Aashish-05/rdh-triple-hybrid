"""Round-trip and validation tests for the explicit-sidecar PEE codec."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.predictors.cnn import CNNPixelPredictor
from src.predictors.rhombus import predict_rhombus
from src.predictors.transformer import WindowTransformerPredictor
from src.rdh.pee import (
    CapacityError,
    SynchronizationError,
    compute_rdh_metrics,
    contract_prediction_error,
    embed_payload,
    expand_prediction_error,
    extract_payload,
    load_sidecar,
    make_rp_predictor,
    make_cnn_predictor,
    make_triple_hybrid_predictor,
    make_transformer_predictor,
    save_marked_image,
    save_sidecar,
)


def _zero_predictor(image: np.ndarray, target_set: str):
    height, width = image.shape
    cross = (np.indices((height, width)).sum(axis=0) % 2) == 0
    target = cross if target_set == "cross" else ~cross
    mask = np.zeros((height, width), dtype=bool)
    if height > 2 and width > 2:
        mask[1:-1, 1:-1] = target[1:-1, 1:-1]
    return np.zeros_like(image), mask


def _underflow_predictor(image: np.ndarray, target_set: str):
    prediction, mask = _zero_predictor(image, target_set)
    target_indices = np.flatnonzero(mask)
    if len(target_indices):
        prediction.reshape(-1)[target_indices[0]] = 1
    return prediction, mask


class TestPredictionErrorMapping(unittest.TestCase):
    def test_zero_and_signed_error_mapping(self):
        self.assertEqual(expand_prediction_error(100, 0, 0), 100)
        self.assertEqual(expand_prediction_error(100, 0, 1), 101)
        self.assertEqual(expand_prediction_error(100, 4), 105)
        self.assertEqual(expand_prediction_error(100, -4), 95)
        self.assertEqual(contract_prediction_error(0, True), (0, 0))
        self.assertEqual(contract_prediction_error(1, True), (0, 1))
        self.assertEqual(contract_prediction_error(5, True), (4, None))
        self.assertEqual(contract_prediction_error(-5, True), (-4, None))
        with self.assertRaises(ValueError):
            expand_prediction_error(100, 0)
        with self.assertRaises(SynchronizationError):
            contract_prediction_error(-1, True)

    def test_overflow_returns_explicit_skip(self):
        self.assertIsNone(expand_prediction_error(255, 0, 1))
        self.assertIsNone(expand_prediction_error(254, 1))
        self.assertIsNone(expand_prediction_error(1, -1))
        self.assertEqual(expand_prediction_error(255, 0, 0), 255)


class TestSidecarRoundTrips(unittest.TestCase):
    def setUp(self):
        self.predictor, self.predictor_metadata = make_rp_predictor()
        self.image = np.full((64, 64), 128, dtype=np.uint8)

    def _round_trip(self, payload: bytes):
        embedded = embed_payload(self.image, payload, self.predictor, self.predictor_metadata)
        extracted = extract_payload(
            embedded.marked_image, embedded.sidecar, self.predictor, self.predictor_metadata
        )
        np.testing.assert_array_equal(extracted.restored_image, self.image)
        self.assertEqual(extracted.payload, payload)
        return embedded

    def test_empty_payload_is_no_op(self):
        embedded = self._round_trip(b"")
        np.testing.assert_array_equal(embedded.marked_image, self.image)
        self.assertEqual(embedded.sidecar.metadata["payload_length_bits"], 0)
        self.assertGreater(embedded.sidecar.metadata["capacity_bits"], 0)

    def test_short_payload_round_trip(self):
        embedded = self._round_trip(b"\xA5")
        self.assertGreater(embedded.sidecar.metadata["pass_counts"]["cross"]["embedded_bits"], 0)

    def test_larger_payload_and_metrics(self):
        payload = bytes(range(128))
        embedded = self._round_trip(payload)
        metrics = compute_rdh_metrics(self.image, embedded.marked_image, embedded.sidecar)
        self.assertEqual(metrics["payload_bits"], len(payload) * 8)
        self.assertGreater(metrics["capacity_bits"], metrics["payload_bits"])
        self.assertGreater(metrics["embedding_rate_bpp"], 0.0)
        self.assertTrue(np.isfinite(metrics["marked_image_psnr_db"]))
        self.assertGreater(metrics["sidecar_bytes"], 0)
        self.assertGreaterEqual(metrics["embedding_runtime_seconds"], 0.0)
        self.assertEqual(
            metrics["predictor_mae_gray_levels"],
            embedded.sidecar.metadata["predictor_mae_gray_levels"],
        )

    def test_boundary_overflow_is_recorded_and_reversible(self):
        image = np.zeros((16, 16), dtype=np.uint8)
        image[2, 2] = 255
        embedded = embed_payload(image, b"\x00", _zero_predictor, {"kind": "zero"})
        self.assertGreater(len(embedded.sidecar.cross_overflow), 0)
        extracted = extract_payload(embedded.marked_image, embedded.sidecar, _zero_predictor, {"kind": "zero"})
        np.testing.assert_array_equal(extracted.restored_image, image)
        self.assertEqual(extracted.payload, b"\x00")

        underflow_image = np.zeros((16, 16), dtype=np.uint8)
        underflow = embed_payload(underflow_image, b"\x00", _underflow_predictor, {"kind": "underflow"})
        self.assertGreater(len(underflow.sidecar.cross_overflow), 0)
        restored = extract_payload(
            underflow.marked_image, underflow.sidecar, _underflow_predictor, {"kind": "underflow"}
        )
        np.testing.assert_array_equal(restored.restored_image, underflow_image)
        self.assertEqual(restored.payload, b"\x00")

        full_white = np.full((16, 16), 255, dtype=np.uint8)
        with self.assertRaises(CapacityError):
            embed_payload(full_white, b"\x00", _zero_predictor, {"kind": "zero"})

    def test_payload_round_trip_on_boundary_values(self):
        image = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
        embedded = self._round_trip_for(image, b"\xF0\x0F")
        self.assertEqual(embedded.sidecar.metadata["payload_length_bits"], 16)

    def _round_trip_for(self, image: np.ndarray, payload: bytes):
        embedded = embed_payload(image, payload, self.predictor, self.predictor_metadata)
        extracted = extract_payload(
            embedded.marked_image, embedded.sidecar, self.predictor, self.predictor_metadata
        )
        np.testing.assert_array_equal(extracted.restored_image, image)
        self.assertEqual(extracted.payload, payload)
        return embedded

    def test_both_passes_and_reverse_order_are_recorded(self):
        embedded = self._round_trip(b"\x01" * 250)
        counts = embedded.sidecar.metadata["pass_counts"]
        self.assertGreater(counts["cross"]["embedded_bits"], 0)
        self.assertGreater(counts["dot"]["embedded_bits"], 0)
        self.assertEqual(embedded.sidecar.metadata["embed_pass_order"], ["cross", "dot"])
        self.assertEqual(embedded.sidecar.metadata["extract_restore_order"], ["dot", "cross"])

    def test_invalid_inputs_and_predictor_mismatch(self):
        with self.assertRaises(ValueError):
            embed_payload(np.zeros((8, 8), dtype=np.float32), b"", self.predictor, self.predictor_metadata)
        with self.assertRaises(TypeError):
            embed_payload(self.image, "not bytes", self.predictor, self.predictor_metadata)

        def wrong_mask(image, target_set):
            return np.zeros_like(image), np.zeros(image.shape, dtype=bool)

        with self.assertRaises(SynchronizationError):
            embed_payload(self.image, b"\x00", wrong_mask, {"kind": "bad mask"})

        embedded = embed_payload(self.image, b"\x00", self.predictor, self.predictor_metadata)
        with self.assertRaises(SynchronizationError):
            extract_payload(embedded.marked_image, embedded.sidecar, self.predictor, {"kind": "wrong"})

    def test_sidecar_round_trip_and_no_overwrite(self):
        embedded = self._round_trip(b"\x02")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "message.rdh.npz"
            image_path = Path(tmp) / "marked.png"
            image_size = save_marked_image(embedded.marked_image, image_path)
            self.assertEqual(image_size, image_path.stat().st_size)
            with Image.open(image_path) as saved:
                np.testing.assert_array_equal(np.asarray(saved), embedded.marked_image)
            size = save_sidecar(embedded.sidecar, path)
            self.assertEqual(size, path.stat().st_size)
            loaded = load_sidecar(path)
            np.testing.assert_array_equal(loaded.cross_overflow, embedded.sidecar.cross_overflow)
            np.testing.assert_array_equal(loaded.dot_overflow, embedded.sidecar.dot_overflow)
            extracted = extract_payload(
                embedded.marked_image, loaded, self.predictor, self.predictor_metadata
            )
            np.testing.assert_array_equal(extracted.restored_image, self.image)
            self.assertEqual(extracted.payload, b"\x02")
            with self.assertRaises(FileExistsError):
                save_sidecar(embedded.sidecar, path)
            with self.assertRaises(FileExistsError):
                save_marked_image(embedded.marked_image, image_path)

    def test_empty_payload_checksum_rejects_changed_image(self):
        embedded = self._round_trip(b"")
        changed = embedded.marked_image.copy()
        changed[0, 0] ^= np.uint8(1)
        with self.assertRaises(SynchronizationError):
            extract_payload(changed, embedded.sidecar, self.predictor, self.predictor_metadata)


class TestHybridPredictorRoundTrip(unittest.TestCase):
    def test_single_neural_predictor_adapters_round_trip(self):
        torch.manual_seed(19)
        image = np.random.default_rng(29).integers(0, 256, size=(16, 16), dtype=np.uint8)
        models = [
            (
                CNNPixelPredictor(in_channels=3, hidden_channels=8).eval(),
                make_cnn_predictor,
            ),
            (
                WindowTransformerPredictor(
                    in_channels=3, dim=8, window_size=8, num_heads=2, num_blocks=1
                ).eval(),
                make_transformer_predictor,
            ),
        ]
        for model, factory in models:
            predictor, metadata = factory(model, device="cpu", checkpoint_sha256="test-checkpoint")
            prediction, mask = predictor(image, "cross")
            cover = image.copy()
            cover[mask] = prediction[mask]
            payload = b"\x5A"
            embedded = embed_payload(cover, payload, predictor, metadata)
            decoded = extract_payload(embedded.marked_image, embedded.sidecar, predictor, metadata)
            np.testing.assert_array_equal(decoded.restored_image, cover)
            self.assertEqual(decoded.payload, payload)

    def test_existing_cnn_transformer_router_synchronize(self):
        torch.manual_seed(7)
        cnn = CNNPixelPredictor(in_channels=3, hidden_channels=8).eval()
        tfp = WindowTransformerPredictor(
            in_channels=3, dim=8, window_size=8, num_heads=2, num_blocks=1, mlp_ratio=2.0
        ).eval()
        predictor, metadata = make_triple_hybrid_predictor(cnn, tfp, T1=10.0, T2=50.0, device="cpu")
        image = np.random.default_rng(12).integers(0, 256, size=(16, 16), dtype=np.uint8)
        prediction, eval_mask = predictor(image, "cross")
        altered_targets = image.copy()
        altered_targets[eval_mask] = 255 - altered_targets[eval_mask]
        repeated_prediction, repeated_mask = predictor(altered_targets, "cross")
        np.testing.assert_array_equal(repeated_mask, eval_mask)
        np.testing.assert_array_equal(repeated_prediction[eval_mask], prediction[eval_mask])
        image[eval_mask] = prediction[eval_mask]
        embedded = embed_payload(image, b"\xC3", predictor, metadata)
        extracted = extract_payload(embedded.marked_image, embedded.sidecar, predictor, metadata)
        np.testing.assert_array_equal(extracted.restored_image, image)
        self.assertEqual(extracted.payload, b"\xC3")


if __name__ == "__main__":
    unittest.main()
