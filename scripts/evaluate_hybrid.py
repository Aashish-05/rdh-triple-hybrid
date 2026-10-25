"""
Triple-Hybrid Evaluation Script for RDH Predictor Framework.

Evaluates all 5 predictor configurations on the 19 USC-SIPI test images,
using tuned (T1, T2) thresholds loaded from router_thresholds.json.

Configurations evaluated:
  1. RP-only       — Rhombus Predictor alone (T1=inf, i.e. always RP)
  2. CNN-only       — CNN Predictor alone
  3. TFP-only       — Transformer Predictor alone
  4. RP+CNN (dual)  — Route low-complexity to RP, rest to CNN (T2=inf)
  5. RP+CNN+TFP     — Full triple hybrid with tuned (T1, T2)

IMPORTANT NOTES:
  - All three predictors are always run on every test image.
    Routing selects which predictor's output is used at each pixel AFTER the
    fact. This is a reference evaluation experiment — it does NOT claim an
    inference speedup from routing.
  - Thresholds come from validation data only (never from test images).
  - Timing is wall-clock for each predictor independently (GPU warm-up included).
  - Do NOT implement RDH embedding/extraction here — that is Phase 5.

Outputs (will NOT overwrite existing baseline CSVs):
  - outputs/tables/hybrid_test_results.csv     (5 configs × aggregate metrics)
  - outputs/tables/hybrid_per_image_results.csv (19 images × 5 configs)
  - outputs/tables/router_usage_statistics.csv  (RP%/CNN%/TFP% per image)
  - outputs/figures/hybrid_comparison.png        (grouped bar — MAE)
  - outputs/figures/router_usage_distribution.png (stacked bar — predictor usage)
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config
from src.data.loader import load_grayscale_image
from src.predictors.rhombus import predict_rhombus, compute_prediction_errors
from src.predictors.cnn import CNNPixelPredictor, predict_cnn_image
from src.predictors.transformer import WindowTransformerPredictor, predict_transformer_image
from src.predictors.router import compute_local_rhombus_complexity, compute_routing_map


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_cnn_model(ckpt_path: Path, device: str) -> CNNPixelPredictor:
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", {})
    model = CNNPixelPredictor(
        in_channels=cfg.get("in_channels", 3),
        hidden_channels=cfg.get("hidden_channels", 48),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def load_tfp_model(ckpt_path: Path, device: str) -> WindowTransformerPredictor:
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", {})
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
    return model


def compute_metrics(image: np.ndarray, pred: np.ndarray, eval_mask: np.ndarray) -> dict:
    """Compute MAE, RMSE, P(e=0), error std from prediction and ground-truth."""
    errors = compute_prediction_errors(image, pred, eval_mask)
    n = len(errors)
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    zero_count = int(np.sum(errors == 0))
    zero_rate = float(zero_count / max(1, n) * 100.0)
    err_std = float(np.std(errors))
    return {
        "evaluated_pixels": n,
        "mae": mae,
        "rmse": rmse,
        "zero_count": zero_count,
        "zero_rate_pct": zero_rate,
        "error_std": err_std,
    }


def apply_routing(
    image: np.ndarray,
    rp_pred: np.ndarray,
    cnn_pred: np.ndarray,
    tfp_pred: np.ndarray,
    eval_mask: np.ndarray,
    T1: float,
    T2: float,
) -> tuple:
    """
    Build a composite prediction using the routing map and pre-computed predictor outputs.

    Returns: (pred_image, routing_map, stats)
      - routing_map values: 0=RP, 1=CNN, 2=TFP, -1=border
    """
    c_map = compute_local_rhombus_complexity(image, exclude_border=True)
    routing = compute_routing_map(c_map, T1, T2, eval_mask)

    H, W = image.shape
    pred = np.zeros((H, W), dtype=np.uint8)
    pred[routing == 0] = rp_pred[routing == 0]
    pred[routing == 1] = cnn_pred[routing == 1]
    pred[routing == 2] = tfp_pred[routing == 2]

    total = int(eval_mask.sum())
    count_rp = int((routing == 0).sum())
    count_cnn = int((routing == 1).sum())
    count_tfp = int((routing == 2).sum())
    stats = {
        "rp_pct": round(count_rp / max(1, total) * 100.0, 2),
        "cnn_pct": round(count_cnn / max(1, total) * 100.0, 2),
        "tfp_pct": round(count_tfp / max(1, total) * 100.0, 2),
        "rp_count": count_rp,
        "cnn_count": count_cnn,
        "tfp_count": count_tfp,
    }
    return pred, routing, stats


def timed_inference(fn_call) -> tuple:
    """Run fn_call(), return (result, elapsed_ms). GPU-synchronized if CUDA available."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = fn_call()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return result, (time.perf_counter() - t0) * 1000.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 75)
    print("PHASE 3.5 — Triple-Hybrid Test-Set Evaluation (5 Configurations)")
    print("=" * 75)

    cfg = get_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[*] Device: {device}")

    tables_dir = cfg.paths.tables_dir
    figures_dir = cfg.paths.figures_dir
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load tuned thresholds
    thresholds_path = tables_dir / "router_thresholds.json"
    if not thresholds_path.exists():
        raise FileNotFoundError(
            f"Tuned thresholds not found: {thresholds_path}\n"
            "Please run scripts/tune_router.py first."
        )
    with open(thresholds_path, "r") as f:
        thr = json.load(f)
    T1_tuned = float(thr["T1"])
    T2_tuned = float(thr["T2"]) if thr["T2"] != "inf" else 1e9
    print(f"\n[*] Loaded tuned thresholds: T1={T1_tuned}, T2={T2_tuned if T2_tuned < 1e8 else 'inf'}")
    print(f"    Tuning Val MAE: {thr.get('val_mae', '?'):.4f}")
    print(f"    Note: {thr.get('tuning_note', '')}")

    # Load models
    print("\n[*] Loading checkpoints...")
    cnn_model = load_cnn_model(cfg.paths.checkpoints_dir / "best_cnnp.pt", device)
    tfp_model = load_tfp_model(cfg.paths.checkpoints_dir / "best_tfp.pt", device)
    print("  Models loaded.")

    # Load test split
    with open(cfg.paths.splits_path, "r") as f:
        splits = json.load(f)
    with open(cfg.paths.manifest_path, "r") as f:
        manifest = json.load(f)
    test_filenames: List[str] = splits["test"]
    print(f"\n[*] Test images: {len(test_filenames)} (USC-SIPI)")

    # GPU warm-up
    if device == "cuda":
        print("[*] GPU warm-up (3 dummy forward passes for each model)...")
        dummy = np.zeros((512, 512), dtype=np.uint8)
        for _ in range(3):
            predict_cnn_image(cnn_model, dummy, target_set="both", exclude_border=True, device=device)
            predict_transformer_image(tfp_model, dummy, target_set="both", exclude_border=True, device=device)
        torch.cuda.synchronize()
        print("  Warm-up done.")

    # Per-image results storage
    per_image_rows = []
    routing_stat_rows = []

    # Aggregate accumulators: 5 config × metrics
    configs = ["RP-only", "CNN-only", "TFP-only", "RP+CNN", "RP+CNN+TFP"]
    agg: Dict[str, List] = {c: [] for c in configs}
    agg_time: Dict[str, List] = {c: [] for c in configs}

    print(f"\n[*] Evaluating 5 configurations on {len(test_filenames)} test images...\n")

    for fn in test_filenames:
        file_path = manifest[fn]["absolute_path"]
        img_arr, meta = load_grayscale_image(file_path, target_size=cfg.dataset.target_size)
        H, W = img_arr.shape

        # --- RP (pure CPU) ---
        (rp_pred, eval_mask), rp_time = timed_inference(
            lambda: predict_rhombus(img_arr, target_set="both", exclude_border=True)
        )

        # --- CNN ---
        (cnn_pred, _), cnn_time = timed_inference(
            lambda: predict_cnn_image(cnn_model, img_arr, target_set="both", exclude_border=True, device=device)
        )

        # --- TFP ---
        (tfp_pred, _), tfp_time = timed_inference(
            lambda: predict_transformer_image(tfp_model, img_arr, target_set="both", exclude_border=True, device=device)
        )

        # Metrics for pure baselines
        m_rp = compute_metrics(img_arr, rp_pred, eval_mask)
        m_cnn = compute_metrics(img_arr, cnn_pred, eval_mask)
        m_tfp = compute_metrics(img_arr, tfp_pred, eval_mask)

        # --- RP+CNN dual hybrid (T2 = 1e9 = infinity, no TFP) ---
        rp_cnn_pred, rp_cnn_routing, rp_cnn_stats = apply_routing(
            img_arr, rp_pred, cnn_pred, tfp_pred, eval_mask, T1=T1_tuned, T2=1e9
        )
        m_rp_cnn = compute_metrics(img_arr, rp_cnn_pred, eval_mask)

        # --- RP+CNN+TFP triple hybrid ---
        hybrid_pred, hybrid_routing, hybrid_stats = apply_routing(
            img_arr, rp_pred, cnn_pred, tfp_pred, eval_mask, T1=T1_tuned, T2=T2_tuned
        )
        m_hybrid = compute_metrics(img_arr, hybrid_pred, eval_mask)

        # Accumulate for averaging
        agg["RP-only"].append(m_rp["mae"])
        agg["CNN-only"].append(m_cnn["mae"])
        agg["TFP-only"].append(m_tfp["mae"])
        agg["RP+CNN"].append(m_rp_cnn["mae"])
        agg["RP+CNN+TFP"].append(m_hybrid["mae"])

        agg_time["RP-only"].append(rp_time)
        agg_time["CNN-only"].append(cnn_time)
        agg_time["TFP-only"].append(tfp_time)
        # Dual hybrid shares RP+CNN time (RP already computed)
        agg_time["RP+CNN"].append(rp_time + cnn_time)
        # Triple hybrid requires all 3 predictors
        agg_time["RP+CNN+TFP"].append(rp_time + cnn_time + tfp_time)

        per_image_rows.append({
            "Filename": fn,
            "Dimensions": f"{H}x{W}",
            "Evaluated_Pixels": m_rp["evaluated_pixels"],
            # MAE
            "RP_MAE": round(m_rp["mae"], 4),
            "CNN_MAE": round(m_cnn["mae"], 4),
            "TFP_MAE": round(m_tfp["mae"], 4),
            "RP_CNN_MAE": round(m_rp_cnn["mae"], 4),
            "Hybrid_MAE": round(m_hybrid["mae"], 4),
            # P(e=0)
            "RP_P(e=0)_%": round(m_rp["zero_rate_pct"], 2),
            "CNN_P(e=0)_%": round(m_cnn["zero_rate_pct"], 2),
            "TFP_P(e=0)_%": round(m_tfp["zero_rate_pct"], 2),
            "RP_CNN_P(e=0)_%": round(m_rp_cnn["zero_rate_pct"], 2),
            "Hybrid_P(e=0)_%": round(m_hybrid["zero_rate_pct"], 2),
            # Runtime (independent per-model timings)
            "RP_Time_ms": round(rp_time, 2),
            "CNN_Time_ms": round(cnn_time, 2),
            "TFP_Time_ms": round(tfp_time, 2),
        })

        routing_stat_rows.append({
            "Filename": fn,
            "T1": T1_tuned,
            "T2": T2_tuned if T2_tuned < 1e8 else float("inf"),
            "RP_%": hybrid_stats["rp_pct"],
            "CNN_%": hybrid_stats["cnn_pct"],
            "TFP_%": hybrid_stats["tfp_pct"],
            "RP_Count": hybrid_stats["rp_count"],
            "CNN_Count": hybrid_stats["cnn_count"],
            "TFP_Count": hybrid_stats["tfp_count"],
            "Total_Evaluated": m_rp["evaluated_pixels"],
        })

        # Determine best single predictor for this image
        best_single = min(
            [("RP", m_rp["mae"]), ("CNN", m_cnn["mae"]), ("TFP", m_tfp["mae"])],
            key=lambda x: x[1]
        )[0]

        print(
            f"  {fn:16s} | RP:{m_rp['mae']:5.3f} CNN:{m_cnn['mae']:5.3f} "
            f"TFP:{m_tfp['mae']:5.3f} | "
            f"RP+CNN:{m_rp_cnn['mae']:5.3f} Hybrid:{m_hybrid['mae']:5.3f} "
            f"[Best_Single={best_single}] "
            f"[Route: RP={hybrid_stats['rp_pct']:.0f}% CNN={hybrid_stats['cnn_pct']:.0f}% TFP={hybrid_stats['tfp_pct']:.0f}%]"
        )

    # -----------------------------------------------------------------------
    # Aggregate summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("AGGREGATE RESULTS (19 USC-SIPI Test Images):")
    print("=" * 75)
    print(f"  {'Config':<18} {'Avg MAE':>9} {'Avg Time (ms)':>16}")
    print(f"  {'-'*18} {'-'*9} {'-'*16}")

    agg_rows = []
    for c in configs:
        avg_mae = float(np.mean(agg[c]))
        avg_time = float(np.mean(agg_time[c]))
        print(f"  {c:<18} {avg_mae:>9.4f} {avg_time:>16.2f}")
        agg_rows.append({
            "Config": c,
            "Avg_MAE": round(avg_mae, 4),
            "Avg_Time_ms": round(avg_time, 2),
        })

    print("=" * 75)

    # Honest assessment
    best_config = min(agg_rows, key=lambda r: r["Avg_MAE"])
    print(f"\n  Best configuration by average MAE: {best_config['Config']} (MAE={best_config['Avg_MAE']:.4f})")

    hybrid_avg = next(r["Avg_MAE"] for r in agg_rows if r["Config"] == "RP+CNN+TFP")
    cnn_avg = next(r["Avg_MAE"] for r in agg_rows if r["Config"] == "CNN-only")
    rp_avg = next(r["Avg_MAE"] for r in agg_rows if r["Config"] == "RP-only")

    print(f"\n  Hybrid vs RP-only improvement: {rp_avg - hybrid_avg:+.4f} MAE")
    print(f"  Hybrid vs CNN-only difference:  {cnn_avg - hybrid_avg:+.4f} MAE (positive = hybrid better)")

    if hybrid_avg > cnn_avg:
        print("  NOTE: Triple-hybrid MAE is WORSE than CNN-only on this test set.")
        print("  This is an honest result. The routing may route some pixels away from CNN")
        print("  to TFP when the CNN would have done better on those pixels.")
    else:
        print("  Triple-hybrid equals or beats CNN-only on this test set.")

    # -----------------------------------------------------------------------
    # Save outputs (do NOT overwrite baseline CSVs)
    # -----------------------------------------------------------------------
    df_per_image = pd.DataFrame(per_image_rows)
    per_image_path = tables_dir / "hybrid_per_image_results.csv"
    df_per_image.to_csv(per_image_path, index=False)
    print(f"\n[+] Saved per-image results: {per_image_path}")

    df_agg = pd.DataFrame(agg_rows)
    agg_path = tables_dir / "hybrid_test_results.csv"
    df_agg.to_csv(agg_path, index=False)
    print(f"[+] Saved aggregate results: {agg_path}")

    df_routing = pd.DataFrame(routing_stat_rows)
    routing_path = tables_dir / "router_usage_statistics.csv"
    df_routing.to_csv(routing_path, index=False)
    print(f"[+] Saved routing statistics: {routing_path}")

    # -----------------------------------------------------------------------
    # Figure 1: Grouped bar chart — MAE comparison across 5 configs
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(test_filenames))
    width = 0.16
    colors = ["steelblue", "crimson", "forestgreen", "darkorange", "purple"]
    offsets = [-2 * width, -width, 0, width, 2 * width]
    col_map = {
        "RP-only": "RP_MAE",
        "CNN-only": "CNN_MAE",
        "TFP-only": "TFP_MAE",
        "RP+CNN": "RP_CNN_MAE",
        "RP+CNN+TFP": "Hybrid_MAE",
    }
    for cfg_name, color, offset in zip(configs, colors, offsets):
        ax.bar(x + offset, df_per_image[col_map[cfg_name]], width, label=cfg_name,
               color=color, alpha=0.82)

    ax.set_ylabel("MAE (gray levels)", fontsize=11)
    ax.set_title(
        "Triple-Hybrid Evaluation: All 5 Configurations on 19 USC-SIPI Benchmark Images\n"
        f"(Tuned T1={T1_tuned}, T2={T2_tuned if T2_tuned < 1e8 else 'inf'})",
        fontsize=12, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(test_filenames, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=9, ncol=5, loc="upper right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.45)
    plt.tight_layout()
    comparison_plot = figures_dir / "hybrid_comparison.png"
    plt.savefig(comparison_plot, dpi=200)
    plt.close()
    print(f"[+] Saved hybrid comparison figure: {comparison_plot}")

    # -----------------------------------------------------------------------
    # Figure 2: Stacked bar — Predictor usage distribution per image
    # -----------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    x2 = np.arange(len(test_filenames))
    bar_rp = ax2.bar(x2, df_routing["RP_%"], label="RP", color="steelblue", alpha=0.85)
    bar_cnn = ax2.bar(x2, df_routing["CNN_%"], bottom=df_routing["RP_%"],
                      label="CNN", color="crimson", alpha=0.85)
    bar_tfp = ax2.bar(
        x2, df_routing["TFP_%"],
        bottom=df_routing["RP_%"] + df_routing["CNN_%"],
        label="TFP", color="forestgreen", alpha=0.85
    )
    ax2.set_ylabel("Predictor Usage (%)", fontsize=11)
    ax2.set_title(
        f"Router Pixel Distribution per Image (T1={T1_tuned}, T2={T2_tuned if T2_tuned < 1e8 else 'inf'})",
        fontsize=12, fontweight="bold"
    )
    ax2.set_xticks(x2)
    ax2.set_xticklabels(test_filenames, rotation=45, ha="right", fontsize=8)
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=10)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    routing_plot = figures_dir / "router_usage_distribution.png"
    plt.savefig(routing_plot, dpi=200)
    plt.close()
    print(f"[+] Saved router usage figure: {routing_plot}")

    print("\n[DONE] Phase 3.5 evaluation complete.")


if __name__ == "__main__":
    main()
