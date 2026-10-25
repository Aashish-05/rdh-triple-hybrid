"""Evaluate frozen clean-v2 predictors on the untouched BOSSbase holdout."""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config
from src.data.loader import load_grayscale_image
from src.data.split_manifest import (
    file_sha256,
    load_split_manifest,
    ordered_filename_sha256,
    resolve_split_paths,
    validate_v2_thresholds,
)
from src.predictors.cnn import CNNPixelPredictor, predict_cnn_image
from src.predictors.rhombus import compute_prediction_errors, predict_rhombus
from src.predictors.router import compute_local_rhombus_complexity, compute_routing_map
from src.predictors.transformer import (
    WindowTransformerPredictor,
    predict_transformer_image,
)


CONFIGURATIONS = ("RP-only", "CNN-only", "Transformer-only", "Triple-hybrid")


def load_cnn(checkpoint_path: Path, device: str) -> CNNPixelPredictor:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_config = checkpoint["config"]
    if model_config.get("dataset_version") != "v2":
        raise ValueError(f"CNN checkpoint is not trained on clean v2: {checkpoint_path}")
    model = CNNPixelPredictor(
        in_channels=model_config["in_channels"],
        hidden_channels=model_config["hidden_channels"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


def load_transformer(checkpoint_path: Path, device: str) -> WindowTransformerPredictor:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_config = checkpoint["config"]
    if model_config.get("dataset_version") != "v2":
        raise ValueError(
            f"Transformer checkpoint is not trained on clean v2: {checkpoint_path}"
        )
    model = WindowTransformerPredictor(
        in_channels=model_config["in_channels"],
        dim=model_config["dim"],
        window_size=model_config["window_size"],
        num_heads=model_config["num_heads"],
        num_blocks=model_config["num_blocks"],
        mlp_ratio=model_config["mlp_ratio"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


def synchronized_call(function):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = function()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return result, (time.perf_counter() - start) * 1000.0


def metric_row(
    filename: str,
    configuration: str,
    image: np.ndarray,
    prediction: np.ndarray,
    eval_mask: np.ndarray,
    inference_ms: float,
) -> dict:
    errors = compute_prediction_errors(image, prediction, eval_mask)
    return {
        "Filename": filename,
        "Configuration": configuration,
        "Status": "success",
        "Error": "",
        "Evaluated_Pixels": int(errors.size),
        "MAE_gray": float(np.mean(np.abs(errors))),
        "RMSE_gray": float(np.sqrt(np.mean(errors.astype(np.float64) ** 2))),
        "Zero_Error_Count": int(np.count_nonzero(errors == 0)),
        "Zero_Error_Rate_pct": float(np.mean(errors == 0) * 100.0),
        "Inference_Runtime_ms": inference_ms,
    }


def failure_row(filename: str, configuration: str, error: Exception) -> dict:
    return {
        "Filename": filename,
        "Configuration": configuration,
        "Status": "failed",
        "Error": f"{type(error).__name__}: {error}",
        "Evaluated_Pixels": 0,
        "MAE_gray": np.nan,
        "RMSE_gray": np.nan,
        "Zero_Error_Count": 0,
        "Zero_Error_Rate_pct": np.nan,
        "Inference_Runtime_ms": np.nan,
    }


def main() -> None:
    cfg = get_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir", type=Path, required=True,
        help="Directory containing the clean-v2 CNN and Transformer checkpoints.",
    )
    parser.add_argument(
        "--thresholds", type=Path, required=True,
        help="Frozen router_thresholds.json selected using validation data.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "outputs" / "clean_v2_experiment" / "holdout_evaluation",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Continue an incomplete evaluation after validating its manifest and rows.",
    )
    args = parser.parse_args()

    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite holdout results: {args.output_dir}")
    if args.resume and not args.output_dir.is_dir():
        raise FileNotFoundError(
            f"Cannot resume: holdout output directory does not exist: {args.output_dir}"
        )
    splits_path, manifest_path = resolve_split_paths(cfg, "v2")
    splits, manifest, split_document = load_split_manifest(splits_path, manifest_path)
    if split_document.get("metadata", {}).get("schema_version") != 2:
        raise ValueError("The selected dataset manifest is not clean split schema v2")
    filenames = splits["holdout"]
    if len(filenames) != 5997:
        raise ValueError(f"Expected 5997 active holdout files, found {len(filenames)}")

    with args.thresholds.open("r", encoding="utf-8") as stream:
        thresholds = json.load(stream)
    cnn_path = args.checkpoint_dir / "best_cnnp.pt"
    transformer_path = args.checkpoint_dir / "best_tfp.pt"
    cnn_hash = file_sha256(cnn_path)
    transformer_hash = file_sha256(transformer_path)
    validate_v2_thresholds(
        thresholds, splits["val"], cnn_hash, transformer_hash
    )

    t1 = float(thresholds["T1"])
    t2_value = thresholds["T2"]
    t2 = 1e9 if t2_value == "inf" else float(t2_value)
    if not t1 < t2:
        raise ValueError(f"Invalid frozen router thresholds: T1={t1}, T2={t2}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cnn = load_cnn(cnn_path, device)
    transformer = load_transformer(transformer_path, device)
    experiment = {
        "schema_version": 1,
        "dataset_version": "v2",
        "evaluation_split": "holdout",
        "holdout_count": len(filenames),
        "holdout_filename_sha256": ordered_filename_sha256(filenames),
        "splits_path": str(splits_path),
        "splits_file_sha256": file_sha256(splits_path),
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": file_sha256(manifest_path),
        "thresholds_path": str(args.thresholds),
        "thresholds_sha256": file_sha256(args.thresholds),
        "threshold_selection_split": "val",
        "thresholds": {"T1": t1, "T2": "inf" if t2 == 1e9 else t2},
        "checkpoints": {
            "cnn_path": str(cnn_path),
            "cnn_sha256": cnn_hash,
            "transformer_path": str(transformer_path),
            "transformer_sha256": transformer_hash,
        },
        "device": device,
        "metric_note": (
            "Predictor MAE/RMSE/zero-error rate are prediction metrics, not RDH "
            "distortion, payload capacity, or sidecar overhead."
        ),
        "holdout_usage": (
            "Used only after model checkpoints and thresholds were frozen; not used "
            "for training, checkpoint selection, or threshold tuning."
        ),
    }
    rows = []
    columns = [
        "Filename", "Configuration", "Status", "Error", "Evaluated_Pixels",
        "MAE_gray", "RMSE_gray", "Zero_Error_Count", "Zero_Error_Rate_pct",
        "Inference_Runtime_ms",
    ]
    per_image_path = args.output_dir / "per_image_results.csv"
    manifest_path = args.output_dir / "experiment_manifest.json"
    aggregate_path = args.output_dir / "aggregate_summary.csv"
    completed_filenames = set()
    if args.resume:
        if aggregate_path.exists():
            raise FileExistsError(
                f"Holdout results already have an aggregate summary: {aggregate_path}"
            )
        if not manifest_path.is_file() or not per_image_path.is_file():
            raise FileNotFoundError(
                "Cannot resume without both experiment_manifest.json and "
                "per_image_results.csv"
            )
        with manifest_path.open("r", encoding="utf-8") as stream:
            saved_experiment = json.load(stream)
        mismatches = [
            key for key, value in experiment.items()
            if saved_experiment.get(key) != value
        ]
        if mismatches:
            raise ValueError(
                "Existing holdout run does not match the frozen experiment: "
                + ", ".join(mismatches)
            )
        with per_image_path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        rows_by_filename = {}
        for row in rows:
            rows_by_filename.setdefault(row["Filename"], []).append(row)
        for filename, filename_rows in rows_by_filename.items():
            if filename not in filenames:
                raise ValueError(f"Existing results contain an unknown image: {filename}")
            configurations = [row["Configuration"] for row in filename_rows]
            if (
                len(filename_rows) != len(CONFIGURATIONS)
                or set(configurations) != set(CONFIGURATIONS)
                or any(row["Status"] != "success" for row in filename_rows)
            ):
                raise ValueError(
                    f"Existing results for {filename} are incomplete or include failures"
                )
            completed_filenames.add(filename)
        output_mode = "a"
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        with manifest_path.open("x", encoding="utf-8") as stream:
            json.dump(experiment, stream, indent=2)
        output_mode = "x"

    with per_image_path.open(output_mode, encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        if not args.resume:
            writer.writeheader()
        for index, filename in enumerate(filenames, start=1):
            if filename in completed_filenames:
                continue
            image_rows = []
            try:
                entry = manifest[filename]
                image, _ = load_grayscale_image(
                    entry["absolute_path"], target_size=cfg.dataset.target_size
                )
                (rp_prediction, eval_mask), rp_ms = synchronized_call(
                    lambda: predict_rhombus(
                        image, target_set="both", exclude_border=True
                    )
                )
                (cnn_prediction, _), cnn_ms = synchronized_call(
                    lambda: predict_cnn_image(
                        cnn, image, target_set="both",
                        exclude_border=True, device=device,
                    )
                )
                (transformer_prediction, _), transformer_ms = synchronized_call(
                    lambda: predict_transformer_image(
                        transformer, image, target_set="both",
                        exclude_border=True, device=device,
                    )
                )
                image_rows.extend([
                    metric_row(filename, "RP-only", image, rp_prediction, eval_mask, rp_ms),
                    metric_row(filename, "CNN-only", image, cnn_prediction, eval_mask, cnn_ms),
                    metric_row(
                        filename, "Transformer-only", image, transformer_prediction,
                        eval_mask, transformer_ms,
                    ),
                ])

                route_start = time.perf_counter()
                complexity = compute_local_rhombus_complexity(
                    image, exclude_border=True
                )
                routing = compute_routing_map(complexity, t1, t2, eval_mask)
                hybrid_prediction = np.zeros_like(image, dtype=np.uint8)
                hybrid_prediction[routing == 0] = rp_prediction[routing == 0]
                hybrid_prediction[routing == 1] = cnn_prediction[routing == 1]
                hybrid_prediction[routing == 2] = transformer_prediction[routing == 2]
                route_ms = (time.perf_counter() - route_start) * 1000.0
                image_rows.append(metric_row(
                    filename, "Triple-hybrid", image, hybrid_prediction, eval_mask,
                    rp_ms + cnn_ms + transformer_ms + route_ms,
                ))
            except Exception as error:
                image_rows = [
                    failure_row(filename, configuration, error)
                    for configuration in CONFIGURATIONS
                ]
                print(f"[FAIL] {filename}: {type(error).__name__}: {error}")

            rows.extend(image_rows)
            writer.writerows(image_rows)
            if index % 100 == 0 or index == len(filenames):
                print(f"[{index}/{len(filenames)}] holdout images processed")

    per_image = pd.DataFrame(rows)
    numeric_columns = (
        "Evaluated_Pixels",
        "MAE_gray",
        "RMSE_gray",
        "Zero_Error_Count",
        "Zero_Error_Rate_pct",
        "Inference_Runtime_ms",
    )
    for column in numeric_columns:
        per_image[column] = pd.to_numeric(per_image[column], errors="coerce")
    successful = per_image[per_image["Status"] == "success"]
    aggregate_rows = []
    for configuration in CONFIGURATIONS:
        selected = successful[successful["Configuration"] == configuration]
        aggregate_rows.append({
            "Configuration": configuration,
            "N_Success": len(selected),
            "N_Failed": len(filenames) - len(selected),
            "MAE_gray_Mean": selected["MAE_gray"].mean(),
            "MAE_gray_Std": selected["MAE_gray"].std(),
            "RMSE_gray_Mean": selected["RMSE_gray"].mean(),
            "RMSE_gray_Std": selected["RMSE_gray"].std(),
            "Zero_Error_Rate_pct_Mean": selected["Zero_Error_Rate_pct"].mean(),
            "Zero_Error_Rate_pct_Std": selected["Zero_Error_Rate_pct"].std(),
            "Inference_Runtime_ms_Mean": selected["Inference_Runtime_ms"].mean(),
            "Inference_Runtime_ms_Std": selected["Inference_Runtime_ms"].std(),
        })
    pd.DataFrame(aggregate_rows).to_csv(aggregate_path, index=False, mode="x")

    failure_count = int((per_image["Status"] != "success").sum())
    print(f"Results: {args.output_dir}")
    print(f"Successful configuration rows: {len(successful)} / {len(filenames) * 4}")
    print(f"Failed configuration rows: {failure_count}")
    if failure_count:
        raise RuntimeError(
            f"Holdout evaluation completed with {failure_count} failed configuration rows"
        )


if __name__ == "__main__":
    main()
