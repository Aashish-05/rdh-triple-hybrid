"""Run actual USC-SIPI round-trip RDH benchmarks into a new versioned directory."""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config
from src.data.loader import load_grayscale_image
from src.data.split_manifest import (
    load_split_manifest,
    resolve_split_paths,
    validate_v2_thresholds,
)
from src.predictors.cnn import CNNPixelPredictor
from src.predictors.rhombus import compute_prediction_errors
from src.predictors.transformer import WindowTransformerPredictor
from src.rdh.pee import (
    CapacityError,
    compute_rdh_metrics,
    embed_payload,
    extract_payload,
    load_sidecar,
    make_cnn_predictor,
    make_rp_predictor,
    make_transformer_predictor,
    make_triple_hybrid_predictor,
    save_marked_image,
    save_sidecar,
)


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "rdh_benchmark_v1"
PAYLOAD_BYTES = 8
CONFIG_ORDER = ("RP-only", "CNN-only", "Transformer-only", "Triple-hybrid")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_models(checkpoint_dir: Path, device: str):
    cnn_path = checkpoint_dir / "best_cnnp.pt"
    tfp_path = checkpoint_dir / "best_tfp.pt"
    if not cnn_path.is_file() or not tfp_path.is_file():
        raise FileNotFoundError(f"Required checkpoints missing: {cnn_path} or {tfp_path}")
    cnn_sha = sha256_file(cnn_path)
    tfp_sha = sha256_file(tfp_path)
    cnn_checkpoint = torch.load(cnn_path, map_location=device, weights_only=True)
    tfp_checkpoint = torch.load(tfp_path, map_location=device, weights_only=True)
    cnn_cfg = cnn_checkpoint["config"]
    tfp_cfg = tfp_checkpoint["config"]
    cnn = CNNPixelPredictor(
        in_channels=cnn_cfg["in_channels"],
        hidden_channels=cnn_cfg["hidden_channels"],
    )
    cnn.load_state_dict(cnn_checkpoint["model_state_dict"])
    cnn.to(device).eval()
    tfp = WindowTransformerPredictor(
        in_channels=tfp_cfg["in_channels"],
        dim=tfp_cfg["dim"],
        window_size=tfp_cfg["window_size"],
        num_heads=tfp_cfg["num_heads"],
        num_blocks=tfp_cfg["num_blocks"],
        mlp_ratio=tfp_cfg["mlp_ratio"],
    )
    tfp.load_state_dict(tfp_checkpoint["model_state_dict"])
    tfp.to(device).eval()
    return cnn, tfp, cnn_sha, tfp_sha, cnn_checkpoint, tfp_checkpoint


def make_payload(filename: str, source_sha256: str) -> bytes:
    seed_material = f"rdh-benchmark-v1|{filename}|{source_sha256}".encode("utf-8")
    return hashlib.sha256(seed_material).digest()[:PAYLOAD_BYTES]


def _result_error(filename: str, configuration: str, error: Exception) -> dict:
    return {
        "Filename": filename,
        "Configuration": configuration,
        "Status": "failed",
        "Error_Type": type(error).__name__,
        "Error": str(error),
        "Cover_Exact": False,
        "Payload_Exact": False,
    }


def run_one(
    filename: str,
    source_path: Path,
    source_mode: str,
    source_sha: str,
    cover: np.ndarray,
    payload: bytes,
    configuration: str,
    predictor,
    predictor_metadata: dict,
    artifact_dir: Path,
    output_dir: Path,
) -> dict:
    image_path = artifact_dir / f"{configuration}.png"
    sidecar_path = artifact_dir / f"{configuration}.npz"
    embedding = embed_payload(cover, payload, predictor, predictor_metadata)
    save_marked_image(embedding.marked_image, image_path)
    save_sidecar(embedding.sidecar, sidecar_path)

    with Image.open(image_path) as marked_file:
        marked_image = np.asarray(marked_file).copy()
    loaded_sidecar = load_sidecar(sidecar_path)
    extraction = extract_payload(marked_image, loaded_sidecar, predictor, predictor_metadata)
    cover_exact = (
        extraction.restored_image.shape == cover.shape
        and np.array_equal(extraction.restored_image, cover)
    )
    payload_exact = extraction.payload == payload
    if not cover_exact or not payload_exact:
        raise AssertionError(
            f"Round-trip failed for {filename}/{configuration}: "
            f"cover_exact={cover_exact}, payload_exact={payload_exact}"
        )

    metrics = compute_rdh_metrics(cover, marked_image, loaded_sidecar)
    if int(loaded_sidecar.metadata["payload_length_bits"]) != len(payload) * 8:
        raise AssertionError("Sidecar payload bit count differs from deterministic payload")

    return {
        "Filename": filename,
        "Configuration": configuration,
        "Status": "success",
        "Error_Type": "",
        "Error": "",
        "Source_Path": str(source_path),
        "Source_SHA256": source_sha,
        "Source_Mode": source_mode,
        "Cover_Width": int(cover.shape[1]),
        "Cover_Height": int(cover.shape[0]),
        "Recovered_Width": int(extraction.restored_image.shape[1]),
        "Recovered_Height": int(extraction.restored_image.shape[0]),
        "Cover_Exact": cover_exact,
        "Payload_Exact": payload_exact,
        "Payload_SHA256": hashlib.sha256(payload).hexdigest(),
        "Payload_Bytes": len(payload),
        "Payload_Bits": metrics["payload_bits"],
        "Capacity_Bits": metrics["capacity_bits"],
        "Capacity_bpp": metrics["capacity_bpp"],
        "Embedding_Rate_bpp": metrics["embedding_rate_bpp"],
        "Capacity_Occupancy": metrics["capacity_occupancy"],
        "Marked_PSNR_dB": metrics["marked_image_psnr_db"],
        "Embedding_Runtime_s": metrics["embedding_runtime_seconds"],
        "Extraction_Runtime_s": extraction.runtime_seconds,
        "Predictor_MAE_gray": metrics["predictor_mae_gray_levels"],
        "Predictor_MAE_Pixels": metrics["predictor_mae_pixel_count"],
        "Sidecar_Bytes": sidecar_path.stat().st_size,
        "Marked_PNG_Bytes": image_path.stat().st_size,
        "Sidecar_Bits_per_Cover_Pixel": (
            sidecar_path.stat().st_size * 8 / cover.size
        ),
        "Sidecar_to_Marked_File_Ratio": (
            sidecar_path.stat().st_size / image_path.stat().st_size
            if image_path.stat().st_size else math.inf
        ),
        "Marked_Image_Path": str(image_path.relative_to(output_dir)),
        "Sidecar_Path": str(sidecar_path.relative_to(output_dir)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-version", choices=("historical", "v2"), default="historical"
    )
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if args.dataset_version == "v2" and (
        args.checkpoint_dir is None or args.thresholds is None
    ):
        parser.error("v2 benchmark requires explicit --checkpoint-dir and --thresholds")
    output_dir = args.output_dir
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite benchmark output directory: {output_dir}"
        )

    cfg = get_config()
    splits_path, manifest_path = resolve_split_paths(cfg, args.dataset_version)
    if not splits_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Existing dataset split/manifest required for benchmark")
    thresholds_path = args.thresholds or cfg.paths.tables_dir / "router_thresholds.json"
    if not thresholds_path.is_file():
        raise FileNotFoundError(f"Validated router thresholds not found: {thresholds_path}")

    splits, manifest, split_document = load_split_manifest(
        splits_path, manifest_path
    )
    if args.dataset_version == "v2" and split_document.get("metadata", {}).get(
        "schema_version"
    ) != 2:
        raise ValueError("The selected dataset split does not declare schema version 2")
    with thresholds_path.open("r", encoding="utf-8") as stream:
        thresholds = json.load(stream)
    test_filenames = splits["test"]
    if len(test_filenames) != 19:
        raise ValueError(f"Expected the existing 19-image USC-SIPI test split, got {len(test_filenames)}")
    if args.dataset_version == "historical" and (
        thresholds.get("T1") != 1.0 or thresholds.get("T2") != 8.0
    ):
        raise ValueError("Saved tuned router thresholds differ from the verified T1=1.0, T2=8.0")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir = args.checkpoint_dir or cfg.paths.checkpoints_dir
    cnn, tfp, cnn_sha, tfp_sha, cnn_checkpoint, tfp_checkpoint = load_models(
        checkpoint_dir, device
    )
    if args.dataset_version == "v2":
        validate_v2_thresholds(thresholds, splits["val"], cnn_sha, tfp_sha)
        if (
            cnn_checkpoint.get("config", {}).get("dataset_version") != "v2"
            or tfp_checkpoint.get("config", {}).get("dataset_version") != "v2"
        ):
            raise ValueError("Both benchmark checkpoints must declare dataset_version v2")
    t1 = float(thresholds["T1"])
    t2_value = thresholds["T2"]
    t2_is_infinite = t2_value == "inf"
    t2 = 1e9 if t2_is_infinite else float(t2_value)
    if not t1 < t2:
        raise ValueError(f"Invalid router thresholds: T1={t1}, T2={t2}")

    predictor_configs = {
        "RP-only": make_rp_predictor(),
        "CNN-only": make_cnn_predictor(cnn, device, cnn_sha),
        "Transformer-only": make_transformer_predictor(tfp, device, tfp_sha),
        "Triple-hybrid": make_triple_hybrid_predictor(
            cnn,
            tfp,
            t1,
            t2,
            device,
            cnn_checkpoint_sha256=cnn_sha,
            tfp_checkpoint_sha256=tfp_sha,
        ),
    }

    prepared = []
    for filename in test_filenames:
        entry = manifest.get(filename)
        if not entry:
            raise KeyError(f"Test image absent from manifest: {filename}")
        source_path = Path(entry["absolute_path"])
        if not source_path.is_file():
            raise FileNotFoundError(f"USC-SIPI benchmark source is missing: {source_path}")
        cover, image_metadata = load_grayscale_image(
            source_path, target_size=cfg.dataset.target_size
        )
        prepared.append((filename, source_path, image_metadata, cover, sha256_file(source_path)))

    output_dir.mkdir(parents=True, exist_ok=False)
    artifact_root = output_dir / "artifacts"
    artifact_root.mkdir()
    config = {
        "schema_version": 1,
        "project": "Reversible Data Hiding: Triple-Hybrid Predictor Framework",
        "dataset_version": args.dataset_version,
        "benchmark": f"existing 19-image USC-SIPI test split from {splits_path}",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": device,
        "predictors": {
            name: metadata for name, (_, metadata) in predictor_configs.items()
        },
        "thresholds": {
            "T1": t1,
            "T2": "inf" if t2_is_infinite else t2,
            "T2_runtime_sentinel": 1e9 if t2_is_infinite else None,
        },
        "checkpoints": {
            "cnn_path": str(checkpoint_dir / "best_cnnp.pt"),
            "cnn_sha256": cnn_sha,
            "cnn_epoch": cnn_checkpoint.get("epoch"),
            "transformer_path": str(checkpoint_dir / "best_tfp.pt"),
            "transformer_sha256": tfp_sha,
            "transformer_epoch": tfp_checkpoint.get("epoch"),
        },
        "payload_policy": (
            f"{PAYLOAD_BYTES}-byte deterministic payload per source image: "
            "first 8 bytes of SHA256(UTF8('rdh-benchmark-v1|' + filename + '|' + source SHA256)); "
            "same payload per image across all configurations"
        ),
        "image_domain": (
            "8-bit grayscale cover returned by existing load_grayscale_image; RGB TIFF source "
            "is converted to grayscale before embedding. Exact recovery is asserted in this "
            "grayscale cover domain; original RGB channel recovery is not claimed."
        ),
        "rdh_protocol": {
            "codec": "project checkerboard PEE implementation; not a verified reproduction of Zhao et al.",
            "embed_order": ["cross", "dot"],
            "extract_restore_order": ["dot", "cross"],
            "infinite_T2_encoding": "1e9 dual-hybrid sentinel used by the existing router",
            "excluded_border_pixels": 1,
            "sidecar_required": True,
            "predictor_mae_is_not_rdh_performance": True,
        },
        "metric_definitions": {
            "capacity_bits": "safe zero-error positions counted during this actual payload embedding, both passes",
            "capacity_bpp": "capacity_bits / grayscale cover pixel count",
            "embedding_rate_bpp": "payload_bits / grayscale cover pixel count",
            "marked_psnr_db": "8-bit PSNR between grayscale cover and marked grayscale image",
            "predictor_mae_gray": "mean absolute signed prediction error over both pre-expansion passes",
            "sidecar_storage": "actual saved compressed NPZ bytes, separate from marked PNG",
            "timings": "wall-clock encode/decode operation timing; disk writes/image decoding excluded",
        },
        "commands": [
            r".\.venv\Scripts\python.exe scripts\check_leakage.py --versioned v2",
            r".\.venv\Scripts\python.exe scripts\evaluate_rdh_benchmark_v1.py --dataset-version v2 --checkpoint-dir checkpoints\clean_v2 --thresholds outputs\clean_v2_experiment\router_tuning\router_thresholds.json --output-dir outputs\clean_v2_experiment\usc_sipi_rdh_v1",
            r".\.venv\Scripts\python.exe -m pytest -q",
        ],
        "source_images": [
            {
                "filename": filename,
                "absolute_path": str(source_path),
                "source_sha256": source_sha,
                "source_mode": metadata["orig_mode"],
                "source_size": list(metadata["orig_size"]),
                "cover_shape": list(cover.shape),
                "cover_sha256": hashlib.sha256(cover.tobytes()).hexdigest(),
            }
            for filename, source_path, metadata, cover, source_sha in prepared
        ],
    }

    rows = []
    for image_index, (filename, source_path, image_metadata, cover, source_sha) in enumerate(prepared, 1):
        payload = make_payload(filename, source_sha)
        image_artifact_dir = artifact_root / Path(filename).stem
        image_artifact_dir.mkdir()
        print(f"[{image_index}/{len(prepared)}] {filename}, {cover.shape[1]}x{cover.shape[0]}")
        for configuration in CONFIG_ORDER:
            predictor, predictor_metadata = predictor_configs[configuration]
            try:
                row = run_one(
                    filename,
                    source_path,
                    image_metadata["orig_mode"],
                    source_sha,
                    cover,
                    payload,
                    configuration,
                    predictor,
                    predictor_metadata,
                    image_artifact_dir,
                    output_dir,
                )
                rows.append(row)
                print(
                    f"  {configuration}: capacity={row['Capacity_Bits']} bits, "
                    f"payload={row['Payload_Bits']} bits, PSNR={row['Marked_PSNR_dB']:.3f} dB, "
                    f"exact cover/payload={row['Cover_Exact']}/{row['Payload_Exact']}"
                )
            except Exception as exc:
                rows.append(_result_error(filename, configuration, exc))
                print(f"  {configuration}: FAILED: {type(exc).__name__}: {exc}")

    per_image = pd.DataFrame(rows)
    per_image.to_csv(output_dir / "per_image_results.csv", index=False, mode="x")
    numeric_metrics = [
        "Capacity_Bits",
        "Capacity_bpp",
        "Payload_Bits",
        "Embedding_Rate_bpp",
        "Capacity_Occupancy",
        "Marked_PSNR_dB",
        "Embedding_Runtime_s",
        "Extraction_Runtime_s",
        "Predictor_MAE_gray",
        "Sidecar_Bytes",
        "Sidecar_Bits_per_Cover_Pixel",
        "Sidecar_to_Marked_File_Ratio",
    ]
    summaries = []
    for configuration in CONFIG_ORDER:
        successful = per_image[
            (per_image["Configuration"] == configuration)
            & (per_image["Status"] == "success")
        ]
        summary = {
            "Configuration": configuration,
            "Images_Attempted": int((per_image["Configuration"] == configuration).sum()),
            "Images_Succeeded": int(len(successful)),
            "Images_Failed": int((per_image["Configuration"] == configuration).sum() - len(successful)),
        }
        for metric in numeric_metrics:
            values = pd.to_numeric(successful.get(metric, pd.Series(dtype=float)), errors="coerce").dropna()
            summary[f"{metric}_Mean"] = float(values.mean()) if len(values) else np.nan
            summary[f"{metric}_Std"] = float(values.std(ddof=1)) if len(values) > 1 else (0.0 if len(values) == 1 else np.nan)
        summaries.append(summary)
    pd.DataFrame(summaries).to_csv(output_dir / "aggregate_summary.csv", index=False, mode="x")
    with (output_dir / "benchmark_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2, allow_nan=False)

    print(f"Per-image results: {output_dir / 'per_image_results.csv'}")
    print(f"Aggregate summary: {output_dir / 'aggregate_summary.csv'}")
    print(f"Configuration manifest: {output_dir / 'benchmark_manifest.json'}")
    if any(row["Status"] != "success" for row in rows):
        raise RuntimeError("One or more benchmark configurations failed; see per_image_results.csv")


if __name__ == "__main__":
    main()
