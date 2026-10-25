"""
Evaluation Script for CNN-Based Pixel Predictor and Comparison with RP Baseline.

Evaluates the trained CNN model on the exact same 19 USC-SIPI test images
and identical 260,100 interior pixels as the RP baseline.

Outputs:
- outputs/tables/cnn_baseline.csv
- outputs/tables/rp_vs_cnn.csv
- outputs/figures/rp_vs_cnn_comparison.png
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
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config
from src.data.loader import load_grayscale_image
from src.predictors.cnn import CNNPixelPredictor, predict_cnn_image
from src.predictors.rhombus import compute_prediction_errors


def main():
    cfg = get_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    figures_dir = cfg.paths.figures_dir
    tables_dir = cfg.paths.tables_dir
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = cfg.paths.checkpoints_dir / "best_cnnp.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Please run scripts/train_cnn.py first.")

    print(f"[*] Loading CNN checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = CNNPixelPredictor(in_channels=3, hidden_channels=48)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"[*] Checkpoint loaded successfully (Epoch {checkpoint.get('epoch', '?')}, Val Loss: {checkpoint.get('val_loss', 0.0):.4f})")

    # Load RP baseline table for paired comparison
    rp_table_path = tables_dir / "rp_baseline.csv"
    if not rp_table_path.exists():
        raise FileNotFoundError(f"RP baseline table not found: {rp_table_path}. Please run evaluate_rp.py first.")
    df_rp = pd.read_csv(rp_table_path).set_index("Filename")

    with open(cfg.paths.splits_path, "r") as f:
        splits = json.load(f)
    with open(cfg.paths.manifest_path, "r") as f:
        manifest = json.load(f)

    test_filenames = splits["test"]
    print(f"[*] Evaluating CNN on {len(test_filenames)} test images (Device: {device})...")

    cnn_results = []
    comparison_results = []

    for fn in test_filenames:
        file_path = manifest[fn]["absolute_path"]
        img_arr, meta = load_grayscale_image(file_path, target_size=cfg.dataset.target_size)
        H, W = img_arr.shape

        # Measure CNN inference time
        start_time = time.perf_counter()
        pred_img, eval_mask = predict_cnn_image(model, img_arr, target_set="both", exclude_border=True, device=device)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        errors = compute_prediction_errors(img_arr, pred_img, eval_mask)
        num_eval = len(errors)
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        zero_count = int(np.sum(errors == 0))
        zero_rate = float(zero_count / num_eval * 100.0)
        std_err = float(np.std(errors))

        cnn_results.append({
            "Filename": fn,
            "Dimensions": f"{H}x{W}",
            "Original_Mode": meta["orig_mode"],
            "Evaluated_Pixels": num_eval,
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "Zero_Error_Count": zero_count,
            "Zero_Error_Rate_%": round(zero_rate, 2),
            "Error_Std": round(std_err, 4),
            "Runtime_ms": round(elapsed_ms, 2),
        })

        # Paired comparison with RP
        rp_mae = df_rp.loc[fn, "MAE"]
        rp_rmse = df_rp.loc[fn, "RMSE"]
        rp_zero_rate = df_rp.loc[fn, "Zero_Error_Rate_%"]
        rp_time = df_rp.loc[fn, "Runtime_ms"]

        # Positive delta_mae means CNN reduced MAE compared to RP
        delta_mae = rp_mae - mae
        pct_mae_reduction = (delta_mae / rp_mae) * 100.0
        delta_zero_rate = zero_rate - rp_zero_rate

        comparison_results.append({
            "Filename": fn,
            "RP_MAE": round(rp_mae, 4),
            "CNN_MAE": round(mae, 4),
            "Delta_MAE": round(delta_mae, 4),
            "MAE_Improvement_%": round(pct_mae_reduction, 2),
            "RP_P(e=0)_%": round(rp_zero_rate, 2),
            "CNN_P(e=0)_%": round(zero_rate, 2),
            "Delta_P(e=0)_%": round(delta_zero_rate, 2),
            "RP_Time_ms": round(rp_time, 2),
            "CNN_Time_ms": round(elapsed_ms, 2),
        })

        print(f"    - {fn:15s} | RP MAE: {rp_mae:5.3f} -> CNN MAE: {mae:5.3f} ({pct_mae_reduction:+5.2f}%) | Time: {elapsed_ms:5.1f}ms")

    # Save cnn_baseline.csv
    df_cnn = pd.DataFrame(cnn_results)
    cnn_csv_path = tables_dir / "cnn_baseline.csv"
    df_cnn.to_csv(cnn_csv_path, index=False)
    print(f"\n[+] Saved CNN baseline table: {cnn_csv_path}")

    # Save rp_vs_cnn.csv
    df_comp = pd.DataFrame(comparison_results)
    comp_csv_path = tables_dir / "rp_vs_cnn.csv"
    df_comp.to_csv(comp_csv_path, index=False)
    print(f"[+] Saved RP vs CNN comparison table: {comp_csv_path}")

    # Summary
    print("=" * 70)
    print("BENCHMARK COMPARISON SUMMARY (19 Images):")
    print(f"  Average RP MAE:          {df_comp['RP_MAE'].mean():.4f}")
    print(f"  Average CNN MAE:         {df_comp['CNN_MAE'].mean():.4f}")
    print(f"  Mean MAE Reduction:      {df_comp['Delta_MAE'].mean():.4f} ({df_comp['MAE_Improvement_%'].mean():+.2f}%)")
    print(f"  Average RP P(e=0):       {df_comp['RP_P(e=0)_%'].mean():.2f}%")
    print(f"  Average CNN P(e=0):      {df_comp['CNN_P(e=0)_%'].mean():.2f}%")
    print(f"  Average CNN Runtime:     {df_comp['CNN_Time_ms'].mean():.2f} ms")
    print("=" * 70)

    # Plot Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(14, 6))
    x_pos = np.arange(len(df_comp))
    width = 0.35

    ax.bar(x_pos - width/2, df_comp["RP_MAE"], width, label="Rhombus Predictor (RP)", color="steelblue", alpha=0.85)
    ax.bar(x_pos + width/2, df_comp["CNN_MAE"], width, label="CNN Predictor (CNNP)", color="crimson", alpha=0.85)

    ax.set_ylabel("Mean Absolute Error (MAE in gray levels)", fontsize=11)
    ax.set_title("Prediction Error Comparison: Rhombus Predictor vs CNN Predictor on 19 Benchmark Images", fontsize=12, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df_comp["Filename"], rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    plt.tight_layout()
    comp_plot_path = figures_dir / "rp_vs_cnn_comparison.png"
    plt.savefig(comp_plot_path, dpi=200)
    plt.close()
    print(f"[+] Saved comparison plot: {comp_plot_path}")


if __name__ == "__main__":
    main()
