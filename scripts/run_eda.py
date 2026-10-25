"""
Exploratory Data Analysis (EDA) for RDH Triple-Hybrid Predictor Framework.

Generates:
1. outputs/figures/eda_sample_grid.png: Representative grid of Train (BOSSbase) and Test (USC-SIPI) images.
2. outputs/figures/eda_histograms.png: Pixel intensity distributions across training and test images.
3. outputs/figures/eda_texture_complexity.png: Local neighborhood complexity (rhombus variance) distributions.
4. outputs/tables/eda_dataset_summary.csv: Overview of dataset partitions and specifications.
5. outputs/tables/eda_image_statistics.csv: Detailed statistics (mean, std, min, max, entropy, local complexity).
6. outputs/tables/eda_benchmark_complexity_ranking.csv: Benchmark images sorted by local texture complexity.
7. outputs/tables/eda_manifest.json: Exact record of every image filename used in EDA figures and tables.
"""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless plotting
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config
from src.data.loader import load_grayscale_image


def compute_entropy(img_arr: np.ndarray) -> float:
    """Compute Shannon entropy of an 8-bit grayscale image."""
    counts, _ = np.histogram(img_arr.ravel(), bins=256, range=(0, 256))
    probs = counts[counts > 0] / float(img_arr.size)
    return float(-np.sum(probs * np.log2(probs)))


def compute_local_complexity_map(img_arr: np.ndarray) -> np.ndarray:
    """
    Compute local rhombus/cross neighborhood complexity for every pixel:
    Uses standard 4-neighbor absolute difference / variance matching RDH rhombus context:
    delta(i, j) = (|u - d| + |l - r|) / 2 + variance of {u, d, l, r}
    """
    arr = img_arr.astype(np.float64)
    # Pad by 1 with reflection
    padded = np.pad(arr, pad_width=1, mode="reflect")
    u = padded[:-2, 1:-1]
    d = padded[2:, 1:-1]
    l = padded[1:-1, :-2]
    r = padded[1:-1, 2:]

    # Rhombus local gradient energy
    grad_energy = 0.5 * (np.abs(u - d) + np.abs(l - r))
    # Rhombus 4-neighbor standard deviation
    stack = np.stack([u, d, l, r], axis=0)
    std_rhombus = np.std(stack, axis=0)
    
    # Combined local texture complexity metric
    complexity = 0.5 * (grad_energy + std_rhombus)
    return complexity


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

    print("[*] Starting EDA analysis...")
    print(f"    Train size: {len(splits['train'])}, Val size: {len(splits['val'])}, Test size: {len(splits['test'])}")

    # 1. Dataset Overview Table
    summary_data = [
        {"Partition": "Train (BOSSbase)", "Count": len(splits["train"]), "Format": "PGM (P5)", "Dimensions": "512x512", "Mode": "8-bit Grayscale", "Source": "BOSSbase 1.01"},
        {"Partition": "Validation (BOSSbase)", "Count": len(splits["val"]), "Format": "PGM (P5)", "Dimensions": "512x512", "Mode": "8-bit Grayscale", "Source": "BOSSbase 1.01"},
        {"Partition": "Holdout (Locked)", "Count": len(splits["holdout"]), "Format": "PGM (P5)", "Dimensions": "512x512", "Mode": "8-bit Grayscale", "Source": "BOSSbase 1.01"},
        {"Partition": "Benchmark Test (USC-SIPI)", "Count": len(splits["test"]), "Format": "TIFF", "Dimensions": "512x512", "Mode": "Grayscale (BT.601 converted if RGB)", "Source": "USC-SIPI Misc"},
    ]
    df_summary = pd.DataFrame(summary_data)
    summary_csv = tables_dir / "eda_dataset_summary.csv"
    df_summary.to_csv(summary_csv, index=False)
    print(f"[+] Saved dataset summary: {summary_csv}")

    # 2. Select representative images for EDA
    # 8 diverse training images from BOSSbase
    train_sample_names = splits["train"][:8]
    # 8 classic/texture benchmark images from USC-SIPI
    test_sample_names = [
        "4.2.03.tiff",   # Baboon (Complex texture)
        "4.2.05.tiff",   # Airplane (Smooth sky + edges)
        "4.2.06.tiff",   # Sailboat (Water texture + mast)
        "4.2.07.tiff",   # Peppers (Smooth gradients + specular)
        "boat.512.tiff", # Boat (Rigging lines + smooth water)
        "house.tiff",    # House (Architectural edges)
        "5.2.08.tiff",   # Couple (Monochrome portrait)
        "7.1.01.tiff",   # Grass texture (High frequency)
    ]
    # Ensure all test sample files exist in test set
    test_sample_names = [fn for fn in test_sample_names if fn in manifest]

    eda_records = []
    loaded_images = {}
    complexity_maps = {}

    # Load and calculate statistics for all test benchmark images (19 images)
    print("[*] Computing descriptive statistics for all 19 benchmark images...")
    for fn in splits["test"]:
        p = manifest[fn]["absolute_path"]
        arr, meta = load_grayscale_image(p)
        c_map = compute_local_complexity_map(arr)
        entropy = compute_entropy(arr)
        loaded_images[fn] = arr
        complexity_maps[fn] = c_map
        eda_records.append({
            "Split": "Test (USC-SIPI)",
            "Filename": fn,
            "Original_Mode": meta["orig_mode"],
            "Converted_Gray": meta["converted_to_gray"],
            "Mean_Intensity": round(float(arr.mean()), 2),
            "Std_Intensity": round(float(arr.std()), 2),
            "Min_Intensity": int(arr.min()),
            "Max_Intensity": int(arr.max()),
            "Entropy_Bits": round(entropy, 3),
            "Mean_Local_Complexity": round(float(c_map.mean()), 3),
            "Std_Local_Complexity": round(float(c_map.std()), 3),
            "P90_Local_Complexity": round(float(np.percentile(c_map, 90)), 3),
        })

    # Load and calculate statistics for 50 representative train images (for statistical profile)
    print("[*] Computing descriptive statistics for sample of training images...")
    for fn in splits["train"][:50]:
        p = manifest[fn]["absolute_path"]
        arr, meta = load_grayscale_image(p)
        c_map = compute_local_complexity_map(arr)
        entropy = compute_entropy(arr)
        if fn in train_sample_names:
            loaded_images[fn] = arr
            complexity_maps[fn] = c_map
        eda_records.append({
            "Split": "Train (BOSSbase)",
            "Filename": fn,
            "Original_Mode": meta["orig_mode"],
            "Converted_Gray": meta["converted_to_gray"],
            "Mean_Intensity": round(float(arr.mean()), 2),
            "Std_Intensity": round(float(arr.std()), 2),
            "Min_Intensity": int(arr.min()),
            "Max_Intensity": int(arr.max()),
            "Entropy_Bits": round(entropy, 3),
            "Mean_Local_Complexity": round(float(c_map.mean()), 3),
            "Std_Local_Complexity": round(float(c_map.std()), 3),
            "P90_Local_Complexity": round(float(np.percentile(c_map, 90)), 3),
        })

    df_stats = pd.DataFrame(eda_records)
    stats_csv = tables_dir / "eda_image_statistics.csv"
    df_stats.to_csv(stats_csv, index=False)
    print(f"[+] Saved image statistics: {stats_csv}")

    # Benchmark Complexity Ranking Table
    df_benchmark_ranking = df_stats[df_stats["Split"] == "Test (USC-SIPI)"].sort_values(
        by="Mean_Local_Complexity", ascending=True
    ).reset_index(drop=True)
    ranking_csv = tables_dir / "eda_benchmark_complexity_ranking.csv"
    df_benchmark_ranking.to_csv(ranking_csv, index=False)
    print(f"[+] Saved benchmark complexity ranking: {ranking_csv}")

    # 3. Figure 1: Image Samples Grid (4x4 grid: 8 Train + 8 Test)
    print("[*] Generating Figure 1: Sample Images Grid...")
    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    fig.suptitle("Representative Grayscale Images (512x512)\nTop 2 Rows: BOSSbase Train | Bottom 2 Rows: USC-SIPI Benchmark Test", fontsize=14, fontweight="bold")

    display_images = [(fn, "Train", loaded_images[fn]) for fn in train_sample_names] + \
                     [(fn, "Test", loaded_images[fn]) for fn in test_sample_names]

    for ax, (fn, split_lbl, arr) in zip(axes.ravel(), display_images):
        ax.imshow(arr, cmap="gray", vmin=0, vmax=255)
        mean_c = complexity_maps[fn].mean()
        ax.set_title(f"[{split_lbl}] {fn}\nMean: {arr.mean():.1f} | Cplx: {mean_c:.2f}", fontsize=9)
        ax.axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    sample_grid_path = figures_dir / "eda_sample_grid.png"
    plt.savefig(sample_grid_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved Figure 1: {sample_grid_path}")

    # 4. Figure 2: Pixel Intensity Histograms
    print("[*] Generating Figure 2: Intensity Histograms...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Train aggregate histogram (using 50 train images)
    train_pixels = np.concatenate([load_grayscale_image(manifest[fn]["absolute_path"])[0].ravel() for fn in splits["train"][:50]])
    test_pixels = np.concatenate([loaded_images[fn].ravel() for fn in splits["test"]])

    axes[0].hist(train_pixels, bins=64, range=(0, 255), color="royalblue", alpha=0.75, edgecolor="black", density=True)
    axes[0].set_title(f"BOSSbase Training Split (50 images aggregated)\nMean={train_pixels.mean():.1f}, Std={train_pixels.std():.1f}")
    axes[0].set_xlabel("Pixel Intensity [0, 255]")
    axes[0].set_ylabel("Probability Density")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].hist(test_pixels, bins=64, range=(0, 255), color="forestgreen", alpha=0.75, edgecolor="black", density=True)
    axes[1].set_title(f"USC-SIPI Benchmark Test Split (All 19 images)\nMean={test_pixels.mean():.1f}, Std={test_pixels.std():.1f}")
    axes[1].set_xlabel("Pixel Intensity [0, 255]")
    axes[1].set_ylabel("Probability Density")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    hist_path = figures_dir / "eda_histograms.png"
    plt.savefig(hist_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved Figure 2: {hist_path}")

    # 5. Figure 3: Texture Complexity Distribution & Heatmaps
    print("[*] Generating Figure 3: Texture Complexity Distributions...")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Local Rhombus Neighborhood Complexity (Gradient + Variance Energy)\nHigh complexity regions motivate the Transformer predictor", fontsize=14, fontweight="bold")

    # Pick 4 contrasting test images: Smooth, Medium, High Texture
    selected_for_cplx = [
        ("Airplane (4.2.05)", "4.2.05.tiff"),
        ("Boat (boat.512)", "boat.512.tiff"),
        ("Peppers (4.2.07)", "4.2.07.tiff"),
        ("Baboon (4.2.03)", "4.2.03.tiff"),
    ]

    for col, (label, fn) in enumerate(selected_for_cplx):
        arr = loaded_images[fn]
        c_map = complexity_maps[fn]
        
        # Row 0: Original Grayscale
        axes[0, col].imshow(arr, cmap="gray", vmin=0, vmax=255)
        axes[0, col].set_title(f"{label}\nOriginal Grayscale", fontsize=10)
        axes[0, col].axis("off")

        # Row 1: Complexity Heatmap
        im = axes[1, col].imshow(c_map, cmap="inferno", vmin=0, vmax=30)
        axes[1, col].set_title(f"Local Complexity Map\nMean={c_map.mean():.2f}, P90={np.percentile(c_map, 90):.2f}", fontsize=10)
        axes[1, col].axis("off")

    # Add shared colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.3])
    fig.colorbar(im, cax=cbar_ax, label="Complexity Metric")

    cplx_path = figures_dir / "eda_texture_complexity.png"
    plt.savefig(cplx_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved Figure 3: {cplx_path}")

    # 6. Save Manifest of files used in EDA
    manifest_record = {
        "description": "Inventory of images and metrics used in Phase 3.1 EDA outputs",
        "train_samples_figure1": train_sample_names,
        "test_samples_figure1": test_sample_names,
        "test_samples_figure3": [fn for _, fn in selected_for_cplx],
        "total_test_images_analyzed": len(splits["test"]),
        "total_train_images_analyzed": 50,
        "generated_figures": [str(sample_grid_path.name), str(hist_path.name), str(cplx_path.name)],
        "generated_tables": [str(summary_csv.name), str(stats_csv.name), str(ranking_csv.name)],
    }
    manifest_json_path = tables_dir / "eda_manifest.json"
    with open(manifest_json_path, "w") as f:
        json.dump(manifest_record, f, indent=2)
    print(f"[+] Saved EDA manifest record: {manifest_json_path}")

    print("\n[+] Phase 3.1 EDA successfully completed!")


if __name__ == "__main__":
    main()
