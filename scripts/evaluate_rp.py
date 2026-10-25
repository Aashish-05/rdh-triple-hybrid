"""
Evaluation Script for Rhombus Predictor (RP) Baseline.
Evaluates the baseline paper's Rhombus Predictor on all 19 approved USC-SIPI benchmark test images.

Evaluated Metrics:
- Image Name, Dimensions, Format
- Evaluated Pixels Count (Interior pixels: 260,100 pixels per 512x512 image, excluding 1-pixel border)
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- Zero-Error Count and Zero-Error Rate (%)
- Runtime per image (ms)

Saves:
- outputs/tables/rp_baseline.csv
- outputs/figures/rp_error_histograms.png
"""
import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config
from src.data.loader import load_grayscale_image
from src.predictors.rhombus import predict_rhombus, compute_prediction_errors


def main():
    cfg = get_config()
    figures_dir = cfg.paths.figures_dir
    tables_dir = cfg.paths.tables_dir
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    with open(cfg.paths.splits_path, "r") as f:
        splits = json.load(f)
    with open(cfg.paths.manifest_path, "r") as f:
        manifest = json.load(f)

    test_filenames = splits["test"]
    print(f"[*] Starting Rhombus Predictor (RP) evaluation on {len(test_filenames)} benchmark test images...")
    print("    Protocol: 2-pass checkerboard prediction (Cross & Dot sets), excluding 1-pixel border.")

    results = []
    error_dict = {}

    for fn in test_filenames:
        file_path = manifest[fn]["absolute_path"]
        img_arr, meta = load_grayscale_image(file_path, target_size=cfg.dataset.target_size)
        H, W = img_arr.shape

        # Measure predictor runtime
        start_time = time.perf_counter()
        # Predict both Cross and Dot interior pixels (standard 2-pass RDH)
        pred_img, eval_mask = predict_rhombus(img_arr, target_set="both", exclude_border=True)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        errors = compute_prediction_errors(img_arr, pred_img, eval_mask)
        error_dict[fn] = errors

        num_eval = len(errors)
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        zero_count = int(np.sum(errors == 0))
        zero_rate = float(zero_count / num_eval * 100.0)
        std_err = float(np.std(errors))

        # Also compute Cross-only metrics for exact 1-pass comparisons
        _, cross_mask = predict_rhombus(img_arr, target_set="cross", exclude_border=True)
        cross_errors = compute_prediction_errors(img_arr, pred_img, cross_mask)
        cross_mae = float(np.mean(np.abs(cross_errors)))
        cross_zero_rate = float(np.sum(cross_errors == 0) / len(cross_errors) * 100.0)

        results.append({
            "Filename": fn,
            "Dimensions": f"{H}x{W}",
            "Original_Mode": meta["orig_mode"],
            "Evaluated_Pixels": num_eval,
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "Zero_Error_Count": zero_count,
            "Zero_Error_Rate_%": round(zero_rate, 2),
            "Error_Std": round(std_err, 4),
            "Cross_MAE": round(cross_mae, 4),
            "Cross_Zero_Rate_%": round(cross_zero_rate, 2),
            "Runtime_ms": round(elapsed_ms, 2),
        })
        print(f"    - {fn:15s} | MAE: {mae:6.3f} | RMSE: {rmse:6.3f} | P(e=0): {zero_rate:5.2f}% | Time: {elapsed_ms:5.2f}ms")

    df = pd.DataFrame(results)
    csv_path = tables_dir / "rp_baseline.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[+] Saved RP baseline evaluation table: {csv_path}")

    # Summary averages
    avg_mae = df["MAE"].mean()
    avg_rmse = df["RMSE"].mean()
    avg_zero_rate = df["Zero_Error_Rate_%"].mean()
    avg_runtime = df["Runtime_ms"].mean()
    print("=" * 65)
    print(f"BENCHMARK AVERAGE (19 Images):")
    print(f"  Mean MAE:             {avg_mae:.4f}")
    print(f"  Mean RMSE:            {avg_rmse:.4f}")
    print(f"  Mean Zero-Error Rate: {avg_zero_rate:.2f}%")
    print(f"  Mean Runtime:         {avg_runtime:.2f} ms")
    print("=" * 65)

    # Generate Figure: Prediction Error Histograms
    print("[*] Generating Figure: RP Prediction Error Histograms...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Rhombus Predictor (RP) Error Distributions (e = x - x_hat)\nSharper peak at e=0 indicates superior RDH embedding capacity", fontsize=14, fontweight="bold")

    plot_samples = [
        ("4.2.07.tiff", "Peppers (Smooth)", axes[0, 0], "forestgreen"),
        ("4.2.05.tiff", "Airplane (Smooth / Edges)", axes[0, 1], "royalblue"),
        ("boat.512.tiff", "Boat (Medium Texture)", axes[1, 0], "darkorange"),
        ("4.2.03.tiff", "Baboon (High Texture)", axes[1, 1], "crimson"),
    ]

    for fn, label, ax, color in plot_samples:
        errs = error_dict[fn]
        p_zero = np.mean(errs == 0) * 100.0
        mae_val = np.mean(np.abs(errs))
        
        # Focus on central interval [-25, 25] for histogram clarity
        ax.hist(errs, bins=51, range=(-25.5, 25.5), color=color, alpha=0.75, edgecolor="black", density=True)
        ax.set_title(f"{label}\nMAE={mae_val:.2f} | P(e=0)={p_zero:.2f}% | Std={errs.std():.2f}", fontsize=11)
        ax.set_xlabel("Prediction Error e = x - x_hat")
        ax.set_ylabel("Probability Density")
        ax.set_xlim(-25, 25)
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_path = figures_dir / "rp_error_histograms.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved RP error histogram figure: {fig_path}")


if __name__ == "__main__":
    main()
