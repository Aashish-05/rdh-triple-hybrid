"""Prediction-error expansion with an explicit overflow sidecar.

This is a project implementation, not a verified reproduction of a published
RDH algorithm. The marked image is not independently decodable without its
sidecar and the exact predictor configuration recorded there.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping

import numpy as np
import torch
from PIL import Image

from src.data.partition import get_checkerboard_masks
from src.predictors.rhombus import predict_rhombus
from src.predictors.router import predict_triple_hybrid
from src.predictors.cnn import CNNPixelPredictor
from src.predictors.cnn import predict_cnn_image
from src.predictors.transformer import WindowTransformerPredictor
from src.predictors.transformer import predict_transformer_image


Predictor = Callable[[np.ndarray, str], tuple[np.ndarray, np.ndarray]]
_SCHEMA_VERSION = 1
_PASSES = ("cross", "dot")


class CapacityError(ValueError):
    """Raised when a payload exceeds the capacity of the supplied cover."""


class SynchronizationError(ValueError):
    """Raised when marked data, sidecar, or predictor state cannot be synchronized."""


@dataclass(frozen=True)
class RDHSidecar:
    """Serializable auxiliary data required to reverse a marked image."""

    metadata: dict[str, Any]
    cross_overflow: np.ndarray
    dot_overflow: np.ndarray


@dataclass(frozen=True)
class EmbeddingResult:
    marked_image: np.ndarray
    sidecar: RDHSidecar


@dataclass(frozen=True)
class ExtractionResult:
    restored_image: np.ndarray
    payload: bytes
    runtime_seconds: float


def _canonical_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    try:
        encoded = json.dumps(dict(metadata), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError("predictor_metadata must be finite JSON-serializable data") from exc


def make_rp_predictor() -> tuple[Predictor, dict[str, Any]]:
    """Return an RP adapter and the identity/configuration to store in the sidecar."""
    metadata = {
        "kind": "rhombus_predictor",
        "rounding": "floor((neighbor_sum + neighbor_count // 2) / neighbor_count)",
        "border": "excluded",
    }

    def predict(image: np.ndarray, target_set: str) -> tuple[np.ndarray, np.ndarray]:
        return predict_rhombus(image, target_set=target_set, exclude_border=True)

    return predict, metadata


def _model_fingerprint(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def make_cnn_predictor(
    model: CNNPixelPredictor,
    device: str = "cpu",
    checkpoint_sha256: str | None = None,
) -> tuple[Predictor, dict[str, Any]]:
    """Adapt the existing CNN prediction function to the RDH pass interface."""
    metadata = {
        "kind": "cnn_only",
        "device": device,
        "border": "excluded",
        "class": type(model).__qualname__,
        "in_channels": model.in_channels,
        "hidden_channels": model.hidden_channels,
        "weights_sha256": _model_fingerprint(model),
        "checkpoint_sha256": checkpoint_sha256,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }

    def predict(image: np.ndarray, target_set: str) -> tuple[np.ndarray, np.ndarray]:
        return predict_cnn_image(
            model, image, target_set=target_set, exclude_border=True, device=device
        )

    return predict, metadata


def make_transformer_predictor(
    model: WindowTransformerPredictor,
    device: str = "cpu",
    checkpoint_sha256: str | None = None,
) -> tuple[Predictor, dict[str, Any]]:
    """Adapt the existing Transformer prediction function to the RDH pass interface."""
    metadata = {
        "kind": "transformer_only",
        "device": device,
        "border": "excluded",
        "class": type(model).__qualname__,
        "in_channels": model.in_channels,
        "dim": model.dim,
        "window_size": model.window_size,
        "num_blocks": len(model.blocks),
        "num_heads": model.blocks[0].attn.num_heads if model.blocks else None,
        "mlp_ratio": (
            model.blocks[0].mlp[0].out_features / model.dim
            if model.blocks and model.dim
            else None
        ),
        "weights_sha256": _model_fingerprint(model),
        "checkpoint_sha256": checkpoint_sha256,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }

    def predict(image: np.ndarray, target_set: str) -> tuple[np.ndarray, np.ndarray]:
        return predict_transformer_image(
            model, image, target_set=target_set, exclude_border=True, device=device
        )

    return predict, metadata


def make_triple_hybrid_predictor(
    cnn_model: CNNPixelPredictor,
    tfp_model: WindowTransformerPredictor,
    T1: float,
    T2: float,
    device: str = "cpu",
    cnn_checkpoint_sha256: str | None = None,
    tfp_checkpoint_sha256: str | None = None,
) -> tuple[Predictor, dict[str, Any]]:
    """Adapt the existing deterministic triple-hybrid predictor for RDH passes."""
    if not math.isfinite(T1) or not math.isfinite(T2) or T2 <= T1:
        raise ValueError("Hybrid router thresholds must be finite and satisfy T2 > T1")
    if not isinstance(cnn_model, torch.nn.Module) or not isinstance(tfp_model, torch.nn.Module):
        raise TypeError("cnn_model and tfp_model must be PyTorch modules")

    metadata = {
        "kind": "triple_hybrid",
        "T1": float(T1),
        "T2": float(T2),
        "device": device,
        "border": "excluded",
        "cnn": {
            "class": type(cnn_model).__qualname__,
            "in_channels": getattr(cnn_model, "in_channels", None),
            "hidden_channels": getattr(cnn_model, "hidden_channels", None),
            "weights_sha256": _model_fingerprint(cnn_model),
            "checkpoint_sha256": cnn_checkpoint_sha256,
        },
        "transformer": {
            "class": type(tfp_model).__qualname__,
            "in_channels": getattr(tfp_model, "in_channels", None),
            "dim": getattr(tfp_model, "dim", None),
            "window_size": getattr(tfp_model, "window_size", None),
            "num_blocks": len(getattr(tfp_model, "blocks", ())),
            "num_heads": (
                getattr(tfp_model.blocks[0].attn, "num_heads", None)
                if getattr(tfp_model, "blocks", None)
                else None
            ),
            "mlp_ratio": (
                tfp_model.blocks[0].mlp[0].out_features / tfp_model.dim
                if getattr(tfp_model, "blocks", None) and getattr(tfp_model, "dim", 0)
                else None
            ),
            "weights_sha256": _model_fingerprint(tfp_model),
            "checkpoint_sha256": tfp_checkpoint_sha256,
        },
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }

    def predict(image: np.ndarray, target_set: str) -> tuple[np.ndarray, np.ndarray]:
        prediction, eval_mask, _, _ = predict_triple_hybrid(
            image=image,
            cnn_model=cnn_model,
            tfp_model=tfp_model,
            T1=T1,
            T2=T2,
            target_set=target_set,
            exclude_border=True,
            device=device,
        )
        return prediction, eval_mask

    return predict, metadata


def expand_prediction_error(prediction: int, error: int, bit: int | None = None) -> int | None:
    """Return the expanded marked pixel, or None when expansion would overflow."""
    if isinstance(prediction, bool) or not isinstance(prediction, (int, np.integer)) or not 0 <= prediction <= 255:
        raise ValueError("prediction must be an integer in [0, 255]")
    if isinstance(error, bool) or not isinstance(error, (int, np.integer)):
        raise ValueError("error must be an integer")
    pixel = int(prediction) + int(error)
    if not 0 <= pixel <= 255:
        raise ValueError("prediction + error must identify an 8-bit source pixel")

    if error == 0:
        if isinstance(bit, bool) or bit not in (0, 1):
            raise ValueError("a zero prediction error requires payload bit 0 or 1")
        marked = int(prediction) + int(bit)
    else:
        if bit is not None:
            raise ValueError("payload bits are embedded only at zero-error pixels")
        marked = pixel + (1 if error > 0 else -1)
    return marked if 0 <= marked <= 255 else None


def contract_prediction_error(expanded_error: int, read_payload_bit: bool) -> tuple[int, int | None]:
    """Invert one expanded error, optionally interpreting 0/1 as a payload bit."""
    if isinstance(expanded_error, bool) or not isinstance(expanded_error, (int, np.integer)):
        raise ValueError("expanded_error must be an integer")
    value = int(expanded_error)
    if read_payload_bit and value in (0, 1):
        return 0, value
    if value in (0, 1):
        return 0, None
    if value >= 2:
        return value - 1, None
    if value <= -2:
        return value + 1, None
    raise SynchronizationError(f"Expanded error {value} cannot be produced by this PEE mapping")


def _validate_image(image: np.ndarray, label: str) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError(f"{label} must be a 2D uint8 grayscale NumPy array")
    if min(image.shape, default=0) < 1:
        raise ValueError(f"{label} dimensions must be nonzero")
    return image


def _target_mask(shape: tuple[int, int], target_set: str) -> np.ndarray:
    height, width = shape
    cross, dot = get_checkerboard_masks(height, width)
    target = cross if target_set == "cross" else dot
    interior = np.zeros(shape, dtype=bool)
    if height > 2 and width > 2:
        interior[1:-1, 1:-1] = True
    return target & interior


def _run_embedding_pass(
    image: np.ndarray,
    target_set: str,
    predictor: Predictor,
    bits: list[int],
    bit_offset: int,
) -> tuple[np.ndarray, np.ndarray, int, int, int, int]:
    prediction, eval_mask = predictor(image, target_set)
    if not isinstance(prediction, np.ndarray) or prediction.shape != image.shape or prediction.dtype != np.uint8:
        raise SynchronizationError("Predictor must return a same-size uint8 prediction array")
    expected_mask = _target_mask(image.shape, target_set)
    if not isinstance(eval_mask, np.ndarray) or eval_mask.dtype != np.bool_ or not np.array_equal(eval_mask, expected_mask):
        raise SynchronizationError(
            f"Predictor {target_set} evaluation mask must cover exactly the interior checkerboard targets"
        )

    out = image.copy()
    overflow: list[int] = []
    embedded = 0
    capacity = 0
    abs_error_sum = 0
    pixel_count = 0
    flat_in = image.reshape(-1)
    flat_pred = prediction.reshape(-1)
    flat_out = out.reshape(-1)
    for flat_index in np.flatnonzero(expected_mask):
        original = int(flat_in[flat_index])
        pred = int(flat_pred[flat_index])
        error = original - pred
        pixel_count += 1
        abs_error_sum += abs(error)
        if error == 0 and original == 255:
            overflow.append(int(flat_index))
            continue
        if error > 0 and original == 255 or error < 0 and original == 0:
            overflow.append(int(flat_index))
            continue

        if error == 0:
            capacity += 1
            if bit_offset + embedded < len(bits):
                marked = expand_prediction_error(pred, error, bits[bit_offset + embedded])
                if marked is None:
                    overflow.append(int(flat_index))
                    continue
                flat_out[flat_index] = marked
                embedded += 1
        else:
            marked = expand_prediction_error(pred, error)
            if marked is None:
                overflow.append(int(flat_index))
                continue
            flat_out[flat_index] = marked

    return (
        out,
        np.asarray(overflow, dtype=np.uint32),
        embedded,
        capacity,
        abs_error_sum,
        pixel_count,
    )


def _as_bits(payload: bytes) -> list[int]:
    return [(byte >> shift) & 1 for byte in payload for shift in range(7, -1, -1)]


def embed_payload(
    cover: np.ndarray,
    payload: bytes,
    predictor: Predictor,
    predictor_metadata: Mapping[str, Any],
) -> EmbeddingResult:
    """Embed bytes by two checkerboard PEE passes and return marked image + sidecar."""
    cover = _validate_image(cover, "cover")
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    metadata_identity = _canonical_metadata(predictor_metadata)
    start = time.perf_counter()
    bits = _as_bits(payload)

    if not bits:
        # Capacity is measured with the normal two-pass predictor context, while
        # the actual marked image remains an exact no-op for an empty message.
        simulated = cover.copy()
        capacities: dict[str, int] = {}
        abs_error_sum = 0
        pixel_count = 0
        for target_set in _PASSES:
            simulated, _, _, capacity, error_sum, count = _run_embedding_pass(
                simulated, target_set, predictor, [], 0
            )
            capacities[target_set] = capacity
            abs_error_sum += error_sum
            pixel_count += count
        marked = cover.copy()
        overflow_by_pass = {name: np.empty(0, dtype=np.uint32) for name in _PASSES}
        pass_counts = {
            name: {"target_pixels": int(_target_mask(cover.shape, name).sum()), "embedded_bits": 0,
                   "capacity_bits": capacities[name], "overflow_pixels": 0}
            for name in _PASSES
        }
        total_capacity = sum(capacities.values())
    else:
        marked = cover.copy()
        overflow_by_pass = {}
        pass_counts = {}
        capacity_by_pass = {}
        abs_error_sum = 0
        pixel_count = 0
        bit_offset = 0
        for target_set in _PASSES:
            marked, overflow, embedded, capacity, error_sum, count = _run_embedding_pass(
                marked, target_set, predictor, bits, bit_offset
            )
            overflow_by_pass[target_set] = overflow
            capacity_by_pass[target_set] = capacity
            pass_counts[target_set] = {
                "target_pixels": int(_target_mask(cover.shape, target_set).sum()),
                "embedded_bits": embedded,
                "capacity_bits": capacity,
                "overflow_pixels": int(len(overflow)),
            }
            bit_offset += embedded
            abs_error_sum += error_sum
            pixel_count += count
        total_capacity = sum(capacity_by_pass.values())
        if bit_offset != len(bits):
            raise CapacityError(
                f"Payload needs {len(bits)} bits; only {total_capacity} safe zero-error positions are available"
            )

    elapsed = time.perf_counter() - start
    predictor_mae = abs_error_sum / pixel_count if pixel_count else 0.0
    metadata = {
        "schema_version": _SCHEMA_VERSION,
        "algorithm": "checkerboard prediction-error expansion; zero-error payload, signed nonzero shift",
        "sidecar_required": True,
        "image_height": int(cover.shape[0]),
        "image_width": int(cover.shape[1]),
        "image_dtype": "uint8",
        "cover_sha256": hashlib.sha256(cover.tobytes()).hexdigest(),
        "payload_length_bytes": len(payload),
        "payload_length_bits": len(bits),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "embed_pass_order": ["cross", "dot"],
        "extract_restore_order": ["dot", "cross"],
        "pass_counts": pass_counts,
        "capacity_bits": int(total_capacity),
        "capacity_estimation": (
            "simulated two-pass expansion with zero payload; marked image is unchanged"
            if not bits
            else "measured during this two-pass embedding"
        ),
        "predictor_mae_gray_levels": float(predictor_mae),
        "predictor_mae_pixel_count": int(pixel_count),
        "predictor_mae_basis": (
            "mean absolute error over evaluated checkerboard pixels before PEE, "
            "including both passes; empty-payload value is from capacity simulation"
        ),
        "predictor": metadata_identity,
        "runtime_seconds": float(elapsed),
    }
    sidecar = RDHSidecar(
        metadata=metadata,
        cross_overflow=overflow_by_pass["cross"],
        dot_overflow=overflow_by_pass["dot"],
    )
    return EmbeddingResult(marked_image=marked, sidecar=sidecar)


def extract_payload(
    marked_image: np.ndarray,
    sidecar: RDHSidecar,
    predictor: Predictor,
    predictor_metadata: Mapping[str, Any],
) -> ExtractionResult:
    """Extract the payload and restore the original cover using the explicit sidecar."""
    marked_image = _validate_image(marked_image, "marked_image")
    if not isinstance(sidecar, RDHSidecar):
        raise TypeError("sidecar must be an RDHSidecar")
    metadata = sidecar.metadata
    if not isinstance(metadata, dict):
        raise SynchronizationError("Sidecar metadata must be a JSON object")
    if metadata.get("schema_version") != _SCHEMA_VERSION:
        raise SynchronizationError("Unsupported sidecar schema version")
    expected_shape = (metadata.get("image_height"), metadata.get("image_width"))
    if expected_shape != marked_image.shape or metadata.get("image_dtype") != "uint8":
        raise SynchronizationError("Marked image dimensions or dtype do not match sidecar")
    if (
        metadata.get("embed_pass_order") != ["cross", "dot"]
        or metadata.get("extract_restore_order") != ["dot", "cross"]
    ):
        raise SynchronizationError("Unsupported checkerboard pass order in sidecar")
    if _canonical_metadata(predictor_metadata) != metadata.get("predictor"):
        raise SynchronizationError("Predictor/router configuration or model fingerprint differs from sidecar")
    bits_expected = metadata.get("payload_length_bits")
    payload_bytes = metadata.get("payload_length_bytes")
    pass_counts = metadata.get("pass_counts")
    if (
        not isinstance(bits_expected, int)
        or bits_expected < 0
        or bits_expected % 8
        or payload_bytes != bits_expected // 8
        or not isinstance(pass_counts, dict)
        or any(not isinstance(pass_counts.get(name), dict) for name in _PASSES)
    ):
        raise SynchronizationError("Invalid payload length or pass counts in sidecar")
    for name in _PASSES:
        counts = pass_counts[name]
        values = [counts.get(field) for field in ("target_pixels", "embedded_bits", "capacity_bits", "overflow_pixels")]
        if any(not isinstance(value, int) or value < 0 for value in values):
            raise SynchronizationError(f"Invalid {name} pass counts in sidecar")
        expected_target_count = int(_target_mask(marked_image.shape, name).sum())
        overflow = getattr(sidecar, f"{name}_overflow")
        if (
            counts["target_pixels"] != expected_target_count
            or counts["embedded_bits"] > counts["capacity_bits"]
            or counts["capacity_bits"] > counts["target_pixels"]
            or counts["overflow_pixels"] > counts["target_pixels"]
            or not isinstance(overflow, np.ndarray)
            or overflow.ndim != 1
            or overflow.dtype != np.uint32
            or len(overflow) != counts["overflow_pixels"]
            or np.any(overflow >= marked_image.size)
            or len(np.unique(overflow)) != len(overflow)
            or (len(overflow) > 1 and np.any(overflow[1:] <= overflow[:-1]))
            or np.any(~_target_mask(marked_image.shape, name).reshape(-1)[overflow.astype(np.intp)])
        ):
            raise SynchronizationError(f"Invalid {name} overflow map or pixel counts in sidecar")
    if sum(pass_counts[name]["embedded_bits"] for name in _PASSES) != bits_expected:
        raise SynchronizationError("Per-pass payload lengths do not match sidecar payload length")

    start = time.perf_counter()
    if bits_expected == 0:
        if hashlib.sha256(marked_image.tobytes()).hexdigest() != metadata.get("cover_sha256"):
            raise SynchronizationError("Empty-payload marked image does not match the recorded cover checksum")
        if hashlib.sha256(b"").hexdigest() != metadata.get("payload_sha256"):
            raise SynchronizationError("Empty-payload checksum does not match sidecar")
        return ExtractionResult(marked_image.copy(), b"", time.perf_counter() - start)

    work = marked_image.copy()
    bits_by_pass: dict[str, list[int]] = {}
    overflow_by_pass = {
        "cross": sidecar.cross_overflow,
        "dot": sidecar.dot_overflow,
    }
    for target_set in reversed(_PASSES):
        prediction, eval_mask = predictor(work, target_set)
        expected_mask = _target_mask(work.shape, target_set)
        if (
            not isinstance(prediction, np.ndarray)
            or prediction.shape != work.shape
            or prediction.dtype != np.uint8
            or not isinstance(eval_mask, np.ndarray)
            or eval_mask.dtype != np.bool_
            or not np.array_equal(eval_mask, expected_mask)
        ):
            raise SynchronizationError("Predictor output/mask does not match the embedding contract")

        overflow = overflow_by_pass[target_set]
        overflow_mask = np.zeros(work.size, dtype=bool)
        overflow_mask[overflow.astype(np.intp)] = True
        flat_target = np.flatnonzero(expected_mask)

        expected_bits = pass_counts[target_set]["embedded_bits"]
        if not isinstance(expected_bits, int) or not 0 <= expected_bits <= pass_counts[target_set]["capacity_bits"]:
            raise SynchronizationError(f"Invalid {target_set} pass bit count")
        found_bits: list[int] = []
        flat_work = work.reshape(-1)
        flat_pred = prediction.reshape(-1)
        flat_mask = expected_mask.reshape(-1)
        for flat_index in flat_target:
            if overflow_mask[flat_index]:
                continue
            marked_pixel = int(flat_work[flat_index])
            pred = int(flat_pred[flat_index])
            marked_error = marked_pixel - pred
            has_bit = len(found_bits) < expected_bits
            original_error, bit = contract_prediction_error(marked_error, has_bit)
            if bit is not None:
                found_bits.append(bit)
            original_pixel = pred + original_error
            if not 0 <= original_pixel <= 255:
                raise SynchronizationError("Decoded pixel is outside the 8-bit range")
            flat_work[flat_index] = original_pixel
        if len(found_bits) != expected_bits:
            raise SynchronizationError(
                f"{target_set} pass recovered {len(found_bits)} bits; sidecar requires {expected_bits}"
            )
        bits_by_pass[target_set] = found_bits

    payload_bits = bits_by_pass["cross"] + bits_by_pass["dot"]
    payload = bytes(
        sum(payload_bits[offset + bit_index] << (7 - bit_index) for bit_index in range(8))
        for offset in range(0, len(payload_bits), 8)
    )
    if len(payload) != payload_bytes:
        raise SynchronizationError("Extracted payload byte count differs from sidecar")
    if hashlib.sha256(work.tobytes()).hexdigest() != metadata.get("cover_sha256"):
        raise SynchronizationError("Restored cover checksum does not match sidecar")
    if hashlib.sha256(payload).hexdigest() != metadata.get("payload_sha256"):
        raise SynchronizationError("Extracted payload checksum does not match sidecar")
    return ExtractionResult(work, payload, time.perf_counter() - start)


def serialize_sidecar(sidecar: RDHSidecar) -> bytes:
    """Serialize explicit auxiliary information as a compressed, pickle-free NPZ."""
    metadata_bytes = json.dumps(sidecar.metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    buffer = BytesIO()
    np.savez_compressed(
        buffer,
        metadata=np.frombuffer(metadata_bytes, dtype=np.uint8),
        cross_overflow=np.asarray(sidecar.cross_overflow, dtype=np.uint32),
        dot_overflow=np.asarray(sidecar.dot_overflow, dtype=np.uint32),
    )
    return buffer.getvalue()


def save_sidecar(sidecar: RDHSidecar, path: str | Path) -> int:
    """Write a sidecar without replacing an existing artifact; return byte size."""
    destination = Path(path)
    data = serialize_sidecar(sidecar)
    with destination.open("xb") as stream:
        stream.write(data)
    return len(data)


def save_marked_image(marked_image: np.ndarray, path: str | Path) -> int:
    """Save a marked grayscale image as lossless PNG without replacing a file."""
    marked_image = _validate_image(marked_image, "marked_image")
    buffer = BytesIO()
    Image.fromarray(marked_image).save(buffer, format="PNG")
    data = buffer.getvalue()
    with Path(path).open("xb") as stream:
        stream.write(data)
    return len(data)


def load_sidecar(path: str | Path) -> RDHSidecar:
    """Load and validate the NPZ sidecar without enabling pickle."""
    with np.load(Path(path), allow_pickle=False) as archive:
        required = {"metadata", "cross_overflow", "dot_overflow"}
        if set(archive.files) != required:
            raise SynchronizationError("Sidecar archive has an unexpected field set")
        metadata = json.loads(archive["metadata"].tobytes().decode("utf-8"))
        cross = archive["cross_overflow"].copy()
        dot = archive["dot_overflow"].copy()
    return RDHSidecar(metadata=metadata, cross_overflow=cross, dot_overflow=dot)


def compute_rdh_metrics(
    cover: np.ndarray,
    marked_image: np.ndarray,
    sidecar: RDHSidecar,
) -> dict[str, float | int]:
    """Report RDH payload, distortion, prediction and explicit sidecar costs."""
    cover = _validate_image(cover, "cover")
    marked_image = _validate_image(marked_image, "marked_image")
    if not isinstance(sidecar, RDHSidecar):
        raise TypeError("sidecar must be an RDHSidecar")
    if cover.shape != marked_image.shape:
        raise ValueError("cover and marked_image dimensions must match")
    if (
        sidecar.metadata.get("image_height") != cover.shape[0]
        or sidecar.metadata.get("image_width") != cover.shape[1]
    ):
        raise ValueError("sidecar dimensions must match cover and marked image")
    payload_bits = int(sidecar.metadata["payload_length_bits"])
    capacity_bits = int(sidecar.metadata["capacity_bits"])
    difference = cover.astype(np.int16) - marked_image.astype(np.int16)
    mse = float(np.mean(np.square(difference.astype(np.float64))))
    psnr = math.inf if mse == 0.0 else 10.0 * math.log10((255.0 ** 2) / mse)
    sidecar_bytes = len(serialize_sidecar(sidecar))
    pixel_count = cover.size
    return {
        "capacity_bits": capacity_bits,
        "capacity_bpp": capacity_bits / pixel_count,
        "payload_bits": payload_bits,
        "embedding_rate_bpp": payload_bits / pixel_count,
        "capacity_occupancy": payload_bits / capacity_bits if capacity_bits else 0.0,
        "marked_image_psnr_db": psnr,
        "embedding_runtime_seconds": float(sidecar.metadata["runtime_seconds"]),
        "predictor_mae_gray_levels": float(sidecar.metadata["predictor_mae_gray_levels"]),
        "predictor_mae_pixel_count": int(sidecar.metadata["predictor_mae_pixel_count"]),
        "sidecar_bytes": sidecar_bytes,
        "sidecar_bits_per_cover_pixel": sidecar_bytes * 8 / pixel_count,
    }


def predictor_mae(cover: np.ndarray, prediction: np.ndarray, eval_mask: np.ndarray) -> float:
    """Compute predictor-only MAE, separate from marked-image distortion."""
    cover = _validate_image(cover, "cover")
    if not isinstance(prediction, np.ndarray) or prediction.shape != cover.shape or prediction.dtype != np.uint8:
        raise ValueError("prediction must be a same-size uint8 array")
    if not isinstance(eval_mask, np.ndarray) or eval_mask.shape != cover.shape or eval_mask.dtype != np.bool_:
        raise ValueError("eval_mask must be a same-size boolean array")
    if not np.any(eval_mask):
        raise ValueError("eval_mask must include at least one pixel")
    return float(np.mean(np.abs(cover[eval_mask].astype(np.int16) - prediction[eval_mask].astype(np.int16))))
