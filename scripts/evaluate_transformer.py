"""
Evaluation Script for Window-Restricted Transformer Predictor (TFP)
and Comprehensive 3-Way Comparison (RP vs CNN vs Transformer).

Evaluates the trained Transformer on the exact same 19 USC-SIPI images
and identical 260,100 interior pixels as RP and CNN.

Outputs:
- outputs/tables/transformer_baseline.csv
- outputs/tables/rp_cnn_transformer_comparison.csv
- outputs/figures/rp_cnn_transformer_comparison.png
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
from src.predictors.transformer import WindowTransformerPredictor, predict_transformer_image
from src.predictors.rhombus import compute_prediction_errors


def main():
    cfg = get_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    figures_dir = cfg.paths.figures_dir
    tables_dir = cfg.paths.tables_dir
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = cfg.paths.checkpoints_dir / "best_tfp.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Please run scripts/train_transformer.py first.")

    print(f"[*] Loading Transformer checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    ckpt_cfg = checkpoint.get("config", {})
    model = WindowTransformerPredictor(
        in_channels=ckpt_cfg.get("in_channels", 3),
        dim=ckpt_cfg.get("dim", 48),
        window_size=ckpt_cfg.get("window_size", 8),
        num_heads=ckpt_cfg.get("num_heads", 4),
        num_blocks=ckpt_cfg.get("num_blocks", 3),
        mlp_ratio=ckpt_cfg.get("mlp_ratio", 2.0),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"[*] Transformer checkpoint loaded (Epoch {checkpoint.get('epoch', '?')}, Val Loss: {checkpoint.get('val_loss', 0.0):.4f})")

    # Load RP and CNN baseline tables
    rp_table_path = tables_dir / "rp_baseline.csv"
    cnn_table_path = tables_dir / "cnn_baseline.csv"
    if not rp_table_path.exists() or not cnn_table_path.exists():
        raise FileNotFoundError("RP or CNN baseline table missing. Run evaluate_rp.py and evaluate_cnn.py first.")

    df_rp = pd.read_csv(rp_table_path).set_index("Filename")
    df_cnn = pd.read_csv(cnn_table_path).set_index("Filename")

    with open(cfg.paths.splits_path, "r") as f:
        splits = json.load(f)
    with open(cfg.paths.manifest_path, "r") as f:
        manifest = json.load(f)

    test_filenames = splits["test"]

    # GPU Warmup
    if device == "cuda":
        print("[*] Warming up GPU for accurate timing...")
        dummy_img = np.zeros((512, 512), dtype=np.uint8)
        for _ in range(3):
            predict_transformer_image(model, dummy_img, target_set="both", exclude_border=True, device=device)
        torch.cuda.synchronize()

    print(f"[*] Evaluating Transformer on {len(test_filenames)} test images (Device: {device})...")
    print("    Timing Note: Measured using torch.cuda.synchronize() around model execution (excludes disk IO).")

    tfp_results = []
    comparison_results = []

    for fn in test_filenames:
        file_path = manifest[fn]["absolute_path"]
        img_arr, meta = load_grayscale_image(file_path, target_size=cfg.dataset.target_size)
        H, W = img_arr.shape

        if device == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        pred_img, eval_mask = predict_transformer_image(
            model, img_arr, target_set="both", exclude_border=True, device=device
        )

        if device == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        errors = compute_prediction_errors(img_arr, pred_img, eval_mask)
        num_eval = len(errors)
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        zero_count = int(np.sum(errors == 0))
        zero_rate = float(zero_count / num_eval * 100.0)
        std_err = float(np.std(errors))

        tfp_results.append({
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

        rp_mae = df_rp.loc[fn, "MAE"]
        cnn_mae = df_cnn.loc[fn, "MAE"]
        rp_zero = df_rp.loc[fn, "Zero_Error_Rate_%"]
        cnn_zero = df_cnn.loc[fn, "Zero_Error_Rate_%"]
        rp_time = df_rp.loc[fn, "Runtime_ms"]
        cnn_time = df_cnn.loc[fn, "Runtime_ms"]

        comparison_results.append({
            "Filename": fn,
            "RP_MAE": round(rp_mae, 4),
            "CNN_MAE": round(cnn_mae, 4),
            "TFP_MAE": round(mae, 4),
            "Delta_MAE_vs_RP": round(rp_mae - mae, 4),
            "Delta_MAE_vs_CNN": round(cnn_mae - mae, 4),
            "RP_P(e=0)_%": round(rp_zero, 2),
            "CNN_P(e=0)_%": round(cnn_zero, 2),
            "TFP_P(e=0)_%": round(zero_rate, 2),
            "RP_Time_ms": round(rp_time, 2),
            "CNN_Time_ms": round(cnn_time, 2),
            "TFP_Time_ms": round(elapsed_ms, 2),
        })

        print(f"    - {fn:15s} | RP: {rp_mae:5.3f} | CNN: {cnn_mae:5.3f} | TFP: {mae:5.3f} | TFP Time: {elapsed_ms:5.1f}ms")

    # Save transformer_baseline.csv
    df_tfp = pd.DataFrame(tfp_results)
    tfp_csv_path = tables_dir / "transformer_baseline.csv"
    df_tfp.to_csv(tfp_csv_path, index=False)
    print(f"\n[+] Saved Transformer baseline table: {tfp_csv_path}")

    # Save rp_cnn_transformer_comparison.csv
    df_comp = pd.DataFrame(comparison_results)
    comp_csv_path = tables_dir / "rp_cnn_transformer_comparison.csv"
    df_comp.to_csv(comp_csv_path, index=False)
    print(f"[+] Saved 3-Way (RP vs CNN vs TFP) comparison table: {comp_csv_path}")

    # Aggregate summary
    print("=" * 75)
    print("3-WAY BENCHMARK COMPARISON SUMMARY (19 Images):")
    print(f"  Average RP MAE:          {df_comp['RP_MAE'].mean():.4f}")
    print(f"  Average CNN MAE:         {df_comp['CNN_MAE'].mean():.4f}")
    print(f"  Average Transformer MAE: {df_comp['TFP_MAE'].mean():.4f}")
    print(f"  Average RP P(e=0):       {df_comp['RP_P(e=0)_%'].mean():.2f}%")
    print(f"  Average CNN P(e=0):      {df_comp['CNN_P(e=0)_%'].mean():.2f}%")
    print(f"  Average TFP P(e=0):      {df_comp['TFP_P(e=0)_%'].mean():.2f}%")
    print(f"  Average RP Runtime:      {df_comp['RP_Time_ms'].mean():.2f} ms")
    print(f"  Average CNN Runtime:     {df_comp['CNN_Time_ms'].mean():.2f} ms")
    print(f"  Average TFP Runtime:     {df_comp['TFP_Time_ms'].mean():.2f} ms")
    print("=" * 75)

    # 3-Way Grouped Bar Chart
    fig, ax = plt.subplots(figsize=(15, 6))
    x_pos = np.arange(len(df_comp))
    width = 0.25

    ax.bar(x_pos - width, df_comp["RP_MAE"], width, label="Rhombus (RP)", color="steelblue", alpha=0.85)
    ax.bar(x_pos, df_comp["CNN_MAE"], width, label="CNN (CNNP)", color="crimson", alpha=0.85)
    ax.bar(x_pos + width, df_comp["TFP_MAE"], width, label="Transformer (TFP)", color="forestgreen", alpha=0.85)

    ax.set_ylabel("Mean Absolute Error (MAE in gray levels)", fontsize=11)
    ax.set_title("3-Way Predictor Comparison: RP vs CNN vs Window Transformer on 19 Benchmark Images", fontsize=12, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df_comp["Filename"], rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    plt.tight_layout()
    comp_plot_path = figures_dir / "rp_cnn_transformer_comparison.png"
    plt.savefig(comp_plot_path, dpi=200)
    plt.close()
    print(f"[+] Saved 3-way comparison plot: {comp_plot_path}")


if __name__ == "__main__":
    main()
