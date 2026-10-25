"""
Training Script for CNN-Based Pixel Predictor Baseline.

Guarantees:
1. Trains STRICTLY on the 3,500 BOSSbase training images.
2. Validates STRICTLY on the 500 BOSSbase validation images.
3. Locked holdout (6,000 images) and USC-SIPI test images (19 images) are never accessed.
4. Non-finite loss detection and graceful exception handling.
5. Saves best validation model as checkpoints/best_cnnp.pt.
6. Saves training curves to outputs/figures/cnn_training_curve.png.
"""
import sys
import time
import json
import argparse
import random
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config
from src.data.training_data import RDHCheckerboardDataset
from src.data.split_manifest import (
    file_sha256,
    ordered_filename_sha256,
    resolve_split_paths,
    load_split_manifest,
)
from src.predictors.cnn import CNNPixelPredictor


def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute L1 loss strictly over evaluated target pixels."""
    diff = torch.abs(pred - target) * mask.float()
    loss = torch.sum(diff) / (torch.sum(mask.float()) + 1e-8)
    return loss


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        x = batch["input"].to(device)
        y = batch["target"].to(device)
        mask = batch["eval_mask"].to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = masked_l1_loss(pred, y, mask)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss detected: {loss.item()}")

        loss.backward()
        # Gradient clipping for numerical stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def evaluate_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_zero_error = 0
    total_pixels = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["input"].to(device)
            y = batch["target"].to(device)
            mask = batch["eval_mask"].to(device)

            pred = model(x)
            loss = masked_l1_loss(pred, y, mask)

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Zero error calculation
            int_pred = torch.clamp(torch.floor(pred + 0.5), 0, 255)
            err = torch.abs(int_pred - y)
            zero_err = (err == 0) & mask
            total_zero_error += zero_err.sum().item()
            total_pixels += mask.sum().item()

    avg_loss = total_loss / total_samples
    zero_error_rate = (total_zero_error / max(1, total_pixels)) * 100.0
    return avg_loss, zero_error_rate


def seed_everything(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-version", choices=("historical", "v2"), default="historical"
    )
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    cfg = get_config()
    seed = cfg.dataset.seed if args.seed is None else args.seed
    if args.dataset_version == "v2" and (
        args.checkpoint_dir is None or args.output_dir is None
    ):
        parser.error("v2 training requires explicit --checkpoint-dir and --output-dir")

    splits_path, manifest_path = resolve_split_paths(cfg, args.dataset_version)
    splits, _, split_document = load_split_manifest(
        splits_path, manifest_path
    )
    if len(splits["train"]) != 3500 or len(splits["val"]) != 500:
        raise ValueError(
            f"Unexpected {args.dataset_version} partition sizes: "
            f"{len(splits['train'])} train, {len(splits['val'])} val"
        )
    if args.dataset_version == "v2" and split_document.get("metadata", {}).get(
        "schema_version"
    ) != 2:
        raise ValueError("The selected v2 split file does not declare schema version 2")

    checkpoint_dir = args.checkpoint_dir or cfg.paths.checkpoints_dir
    custom_output_dir = args.output_dir is not None
    output_dir = args.output_dir or cfg.paths.figures_dir
    best_ckpt_path = checkpoint_dir / "best_cnnp.pt"
    history_path = (
        output_dir / "cnn_history.json"
        if custom_output_dir
        else checkpoint_dir / "cnn_history.json"
    )
    config_path = (
        output_dir / "cnn_training_config.json"
        if custom_output_dir
        else checkpoint_dir / "cnn_training_config.json"
    )
    fig_path = output_dir / "cnn_training_curve.png"
    existing = [
        path for path in (best_ckpt_path, history_path, config_path, fig_path)
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite CNN training artifacts: "
            + ", ".join(str(path) for path in existing)
        )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if custom_output_dir:
        output_dir.mkdir(parents=True, exist_ok=False)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    generator = seed_everything(seed)
    device = torch.device(cfg.hardware.device if torch.cuda.is_available() else "cpu")
    print(f"[*] Training CNN Predictor on device: {device}")
    if device.type == "cuda":
        print(f"    GPU: {torch.cuda.get_device_name(0)} (4 GB VRAM)")

    print(
        f"[*] Dataset version: {args.dataset_version} | "
        f"{len(splits['train'])} Train | {len(splits['val'])} Val"
    )

    # 2. Setup Dataloaders
    batch_size = cfg.hardware.batch_size  # 16
    crop_size = 128  # Memory-efficient crop
    print(f"[*] Loading Datasets (Batch Size: {batch_size}, Crop Size: {crop_size}x{crop_size})...")

    train_ds = RDHCheckerboardDataset(
        split="train", crop_size=crop_size, seed=seed,
        splits_path=splits_path, manifest_path=manifest_path,
    )
    val_ds = RDHCheckerboardDataset(
        split="val", crop_size=crop_size, seed=seed,
        splits_path=splits_path, manifest_path=manifest_path,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=cfg.hardware.num_workers, generator=generator,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=cfg.hardware.num_workers)

    # 3. Initialize Model & Optimizer
    model = CNNPixelPredictor(in_channels=3, hidden_channels=48).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] CNN Model initialized with {total_params:,} trainable parameters.")

    num_epochs = 5
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_zero_rate": [],
        "learning_rates": [],
    }

    best_val_loss = float("inf")
    start_total_time = time.time()

    print("\n" + "=" * 70)
    print(f"{'Epoch':^7} | {'Train MAE':^12} | {'Val MAE':^12} | {'Val P(e=0)':^12} | {'LR':^10} | {'Time':^8}")
    print("=" * 70)

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_zero_rate = evaluate_epoch(model, val_loader, device)
        scheduler.step()

        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_zero_rate"].append(val_zero_rate)
        history["learning_rates"].append(current_lr)

        is_best = val_loss < best_val_loss
        marker = " (*)" if is_best else ""
        if is_best:
            best_val_loss = val_loss
            # Save best checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_zero_rate": val_zero_rate,
                "config": {
                    "in_channels": 3,
                    "hidden_channels": 48,
                    "crop_size": crop_size,
                    "seed": seed,
                    "dataset_version": args.dataset_version,
                    "train_split_sha256": ordered_filename_sha256(splits["train"]),
                    "val_split_sha256": ordered_filename_sha256(splits["val"]),
                }
            }, best_ckpt_path)

        print(f"{epoch:^7d} | {train_loss:^12.4f} | {val_loss:^12.4f} | {val_zero_rate:^11.2f}% | {current_lr:^10.6f} | {elapsed:^7.1f}s{marker}")

    total_training_time = time.time() - start_total_time
    history["total_training_time_seconds"] = total_training_time
    print("=" * 70)
    print(f"[+] Training completed in {total_training_time:.1f}s.")
    print(f"[+] Best Validation Checkpoint saved to: {best_ckpt_path}")
    print(f"    Best Val Loss (MAE): {best_val_loss:.4f}")

    # 4. Save History JSON
    with open(history_path, "x", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[+] Saved training history: {history_path}")

    training_config = {
        "dataset_version": args.dataset_version,
        "splits_path": str(splits_path),
        "manifest_path": str(manifest_path),
        "splits_file_sha256": file_sha256(splits_path),
        "manifest_file_sha256": file_sha256(manifest_path),
        "train_count": len(splits["train"]),
        "val_count": len(splits["val"]),
        "train_filenames_sha256": ordered_filename_sha256(splits["train"]),
        "val_filenames_sha256": ordered_filename_sha256(splits["val"]),
        "seed": seed,
        "device": str(device),
        "epochs": num_epochs,
        "batch_size": batch_size,
        "crop_size": crop_size,
        "optimizer": "AdamW",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "scheduler": "CosineAnnealingLR",
        "scheduler_t_max": num_epochs,
        "scheduler_eta_min": 1e-5,
        "gradient_clip_norm": 5.0,
        "selection_metric": "minimum validation masked L1",
        "training_time_seconds": total_training_time,
        "checkpoint_path": str(best_ckpt_path),
        "checkpoint_sha256": file_sha256(best_ckpt_path),
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_version": torch.version.cuda,
        "model": {"in_channels": 3, "hidden_channels": 48},
    }
    with config_path.open("x", encoding="utf-8") as stream:
        json.dump(training_config, stream, indent=2)
    print(f"[+] Saved training configuration: {config_path}")

    # 5. Plot Loss Curves
    fig, ax1 = plt.subplots(figsize=(8, 5))
    epochs_range = range(1, num_epochs + 1)
    
    ax1.plot(epochs_range, history["train_loss"], "b-o", label="Train Loss (MAE)", linewidth=2)
    ax1.plot(epochs_range, history["val_loss"], "r--s", label="Val Loss (MAE)", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss (MAE in gray levels)", color="black", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(epochs_range, history["val_zero_rate"], "g-.^", label="Val P(e=0) %", linewidth=2)
    ax2.set_ylabel("Zero-Error Rate P(e=0) (%)", color="green", fontsize=11)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title("CNN Predictor Training Curves (5 Epochs on 3,500 BOSSbase Images)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"[+] Saved training curve figure: {fig_path}")


if __name__ == "__main__":
    main()
