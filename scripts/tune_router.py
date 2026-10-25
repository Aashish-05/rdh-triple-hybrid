"""
Router Threshold Tuning Script for RDH Triple-Hybrid Predictor Framework.

Objective: Find optimal (T1, T2) thresholds that minimise mean MAE on the 500
validation images (BOSSbase), strictly using NO test images.

Strategy:
  1. Load CNN and Transformer models from checkpoints.
  2. Load the 500 validation images from the dataset manifest.
  3. For each validation image, precompute RP, CNN, and TFP predictions ONCE.
  4. Grid-search over candidate (T1, T2) pairs:
       T1 in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
       T2 in [8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 1e9]  (1e9 = infinity = RP+CNN only)
       Constraint: T2 > T1
     For each (T1, T2): apply routing map to the pre-computed predictions and
     compute image-level MAE (no re-running CNN/TFP = fast).
  5. Select the (T1, T2) with minimum mean val MAE.
  6. Save router_thresholds.json and router_tuning_summary.csv.

Design Note — Zero Target Leakage:
  The complexity score is computed only from 4 orthogonal context pixels.
  The decoder can reconstruct the identical routing map without the cover image,
  which is a necessary condition for reversibility in downstream RDH embedding.
"""

import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config
from src.data.split_manifest import (
    file_sha256,
    resolve_split_paths,
    load_split_manifest,
    ordered_filename_sha256,
)
from src.data.loader import load_grayscale_image
from src.predictors.rhombus import predict_rhombus, compute_prediction_errors
from src.predictors.cnn import CNNPixelPredictor, predict_cnn_image
from src.predictors.transformer import WindowTransformerPredictor, predict_transformer_image
from src.predictors.router import compute_local_rhombus_complexity, compute_routing_map


# ---------------------------------------------------------------------------
# Grid-search candidates
# ---------------------------------------------------------------------------
T1_CANDIDATES = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
T2_CANDIDATES = [8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 1e9]  # 1e9 acts as infinity


def load_cnn_model(
    ckpt_path: Path, device: str, require_clean_v2: bool = False
) -> CNNPixelPredictor:
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", {})
    if require_clean_v2 and cfg.get("dataset_version") != "v2":
        raise ValueError(f"CNN checkpoint is not trained on clean v2: {ckpt_path}")
    model = CNNPixelPredictor(
        in_channels=cfg.get("in_channels", 3),
        hidden_channels=cfg.get("hidden_channels", 48),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(
        f"  [CNN] Loaded (Epoch {checkpoint.get('epoch', '?')}, "
        f"Val Loss {checkpoint.get('val_loss', 0.0):.4f})"
    )
    return model


def load_tfp_model(
    ckpt_path: Path, device: str, require_clean_v2: bool = False
) -> WindowTransformerPredictor:
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", {})
    if require_clean_v2 and cfg.get("dataset_version") != "v2":
        raise ValueError(f"Transformer checkpoint is not trained on clean v2: {ckpt_path}")
    model = WindowTransformerPredictor(
        in_channels=cfg.get("in_channels", 3),
        dim=cfg.get("dim", 48),
        window_size=cfg.get("window_size", 8),
        num_heads=cfg.get("num_heads", 4),
        num_blocks=cfg.get("num_blocks", 3),
        mlp_ratio=cfg.get("mlp_ratio", 2.0),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(
        f"  [TFP] Loaded (Epoch {checkpoint.get('epoch', '?')}, "
        f"Val Loss {checkpoint.get('val_loss', 0.0):.4f})"
    )
    return model


def compute_mae_for_config(
    image: np.ndarray,
    rp_pred: np.ndarray,
    cnn_pred: np.ndarray,
    tfp_pred: np.ndarray,
    eval_mask: np.ndarray,
    complexity_map: np.ndarray,
    T1: float,
    T2: float,
) -> float:
    """Apply routing and return MAE for the given (T1, T2) pair. No model inference."""
    routing = compute_routing_map(complexity_map, T1, T2, eval_mask)

    H, W = image.shape
    pred = np.zeros((H, W), dtype=np.uint8)
    pred[routing == 0] = rp_pred[routing == 0]
    pred[routing == 1] = cnn_pred[routing == 1]
    pred[routing == 2] = tfp_pred[routing == 2]

    errors = compute_prediction_errors(image, pred, eval_mask)
    return float(np.mean(np.abs(errors)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-version", choices=("historical", "v2"), default="historical"
    )
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.dataset_version == "v2" and (
        args.checkpoint_dir is None or args.output_dir is None
    ):
        parser.error("v2 tuning requires explicit --checkpoint-dir and --output-dir")

    print("=" * 70)
    print("PHASE 3.5 — Router Threshold Tuning")
    print("=" * 70)

    cfg = get_config()
    splits_path, manifest_path = resolve_split_paths(cfg, args.dataset_version)
    splits, manifest, split_document = load_split_manifest(
        splits_path, manifest_path
    )
    if args.dataset_version == "v2" and split_document.get("metadata", {}).get(
        "schema_version"
    ) != 2:
        raise ValueError("The selected v2 split file does not declare schema version 2")
    if len(splits["val"]) != 500:
        raise ValueError(f"Expected 500 validation images, found {len(splits['val'])}")
    checkpoint_dir = args.checkpoint_dir or cfg.paths.checkpoints_dir
    custom_output_dir = args.output_dir is not None
    tables_dir = args.output_dir or cfg.paths.tables_dir
    output_paths = (
        tables_dir / "router_tuning_summary.csv",
        tables_dir / "router_thresholds.json",
    )
    existing = [path for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite router-tuning artifacts: "
            + ", ".join(str(path) for path in existing)
        )
    if custom_output_dir:
        tables_dir.mkdir(parents=True, exist_ok=False)
    else:
        tables_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[*] Device: {device}")

    # Load models
    print("\n[1/5] Loading model checkpoints...")
    cnn_path = checkpoint_dir / "best_cnnp.pt"
    tfp_path = checkpoint_dir / "best_tfp.pt"
    require_clean_v2 = args.dataset_version == "v2"
    cnn_model = load_cnn_model(cnn_path, device, require_clean_v2)
    tfp_model = load_tfp_model(tfp_path, device, require_clean_v2)

    # Load manifest and val split
    print("\n[2/5] Loading validation split (500 images)...")
    val_filenames = splits["val"]
    print(f"  Found {len(val_filenames)} validation images.")

    # GPU warm-up
    if device == "cuda":
        print("\n[*] GPU warm-up (3 dummy forward passes)...")
        dummy = np.zeros((512, 512), dtype=np.uint8)
        for _ in range(3):
            predict_cnn_image(cnn_model, dummy, target_set="both", exclude_border=True, device=device)
        torch.cuda.synchronize()
        print("  Warm-up done.")

    # Precompute per-image predictions (RP, CNN, TFP) on all 500 val images
    print(f"\n[3/5] Precomputing RP, CNN, TFP predictions for all {len(val_filenames)} val images...")
    print("  (This runs each model once per image. Grid search is cheap after this.)")

    val_data = []
    t_start_precompute = time.perf_counter()

    for idx, fn in enumerate(val_filenames):
        file_path = manifest[fn]["absolute_path"]
        img_arr, _ = load_grayscale_image(file_path, target_size=cfg.dataset.target_size)

        # RP (CPU, very fast)
        rp_pred, eval_mask = predict_rhombus(img_arr, target_set="both", exclude_border=True)

        # CNN
        cnn_pred, _ = predict_cnn_image(
            cnn_model, img_arr, target_set="both", exclude_border=True, device=device
        )

        # TFP
        tfp_pred, _ = predict_transformer_image(
            tfp_model, img_arr, target_set="both", exclude_border=True, device=device
        )

        # Complexity map
        c_map = compute_local_rhombus_complexity(img_arr, exclude_border=True)

        val_data.append({
            "fn": fn,
            "image": img_arr,
            "rp_pred": rp_pred,
            "cnn_pred": cnn_pred,
            "tfp_pred": tfp_pred,
            "eval_mask": eval_mask,
            "c_map": c_map,
        })

        if (idx + 1) % 50 == 0:
            elapsed = time.perf_counter() - t_start_precompute
            eta = elapsed / (idx + 1) * (len(val_filenames) - idx - 1)
            print(f"    {idx+1}/{len(val_filenames)} images done — ETA: {eta:.0f}s")

    precompute_time = time.perf_counter() - t_start_precompute
    print(f"  Precompute complete in {precompute_time:.1f}s.")

    # Grid search
    print("\n[4/5] Running grid search over (T1, T2) pairs...")
    search_results = []
    best_mae = float("inf")
    best_T1 = cfg.predictor.initial_T1
    best_T2 = cfg.predictor.initial_T2

    for T1 in T1_CANDIDATES:
        for T2 in T2_CANDIDATES:
            if T2 <= T1:
                continue  # Constraint: T2 must be strictly > T1

            image_maes = []
            for d in val_data:
                mae = compute_mae_for_config(
                    d["image"], d["rp_pred"], d["cnn_pred"], d["tfp_pred"],
                    d["eval_mask"], d["c_map"], T1, T2,
                )
                image_maes.append(mae)

            mean_mae = float(np.mean(image_maes))
            T2_display = T2 if T2 < 1e8 else float("inf")

            search_results.append({
                "T1": T1,
                "T2": T2_display,
                "mean_val_MAE": round(mean_mae, 6),
            })

            if mean_mae < best_mae:
                best_mae = mean_mae
                best_T1, best_T2 = T1, T2

            mode = "RP+CNN" if T2 >= 1e8 else "RP+CNN+TFP"
            print(f"  T1={T1:4.1f}, T2={'inf':>5} [{mode}]  Val MAE={mean_mae:.4f}"
                  if T2 >= 1e8 else
                  f"  T1={T1:4.1f}, T2={T2:5.1f} [{mode}]  Val MAE={mean_mae:.4f}")

    # Also evaluate pure baselines for comparison (fixed "all-RP", "all-CNN", "all-TFP")
    print("\n  -- Fixed Baselines (no routing) --")
    for label, pred_key in [("RP-only", "rp_pred"), ("CNN-only", "cnn_pred"), ("TFP-only", "tfp_pred")]:
        maes = [
            float(np.mean(np.abs(compute_prediction_errors(d["image"], d[pred_key], d["eval_mask"]))))
            for d in val_data
        ]
        m = float(np.mean(maes))
        print(f"  {label:10s}: Val MAE={m:.4f}")
        search_results.append({
            "T1": label,
            "T2": label,
            "mean_val_MAE": round(m, 6),
        })

    # Save summary CSV
    df_search = pd.DataFrame(search_results)
    summary_csv = tables_dir / "router_tuning_summary.csv"
    df_search.to_csv(summary_csv, index=False)
    print(f"\n[+] Saved tuning summary: {summary_csv}")

    # Save best thresholds JSON
    best_T2_display = best_T2 if best_T2 < 1e8 else float("inf")
    result_json = {
        "T1": best_T1,
        "T2": "inf" if best_T2_display == float("inf") else best_T2_display,
        "val_mae": round(best_mae, 6),
        "objective": "minimize_mean_val_mae_over_500_bossbase_images",
        "dataset_version": args.dataset_version,
        "selection_split": "val",
        "validation_count": len(val_filenames),
        "splits_path": str(splits_path),
        "manifest_path": str(manifest_path),
        "splits_file_sha256": file_sha256(splits_path),
        "manifest_file_sha256": file_sha256(manifest_path),
        "validation_filenames_sha256": ordered_filename_sha256(val_filenames),
        "checkpoints": {
            "cnn_path": str(cnn_path),
            "cnn_sha256": file_sha256(cnn_path),
            "transformer_path": str(tfp_path),
            "transformer_sha256": file_sha256(tfp_path),
        },
        "T1_candidates": T1_CANDIDATES,
        "T2_candidates": [t if t < 1e8 else "inf" for t in T2_CANDIDATES],
        "tuning_note": (
            "Thresholds selected on 500 BOSSbase validation images only. "
            "Test images (USC-SIPI) were NOT used during tuning."
        ),
    }
    thresholds_json = tables_dir / "router_thresholds.json"
    with thresholds_json.open("x", encoding="utf-8") as f:
        json.dump(result_json, f, indent=2)
    print(f"[+] Saved best thresholds: {thresholds_json}")

    print("\n" + "=" * 70)
    print("ROUTER TUNING COMPLETE")
    print(f"  Best T1 = {best_T1}")
    print(f"  Best T2 = {best_T2_display}")
    print(f"  Best Val MAE = {best_mae:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
