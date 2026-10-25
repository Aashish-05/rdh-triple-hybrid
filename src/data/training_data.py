"""
Memory-Efficient Dataset and Patch Pipeline for RDH CNN Training.

Guarantees:
1. Strictly references only approved splits (3,500 train, 500 val).
2. On-the-fly random cropping (e.g. 128x128) - never loads entire dataset into RAM.
3. Precomputes Rhombus prior on-the-fly to enforce exact zero-leakage checkerboard context.
"""
from typing import Optional
import random
import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import get_config
from src.data.loader import load_grayscale_image
from src.data.split_manifest import load_split_manifest
from src.predictors.rhombus import predict_rhombus


class RDHCheckerboardDataset(Dataset):
    """
    On-the-fly Dataset for training CNN and Transformer predictors.
    """
    def __init__(
        self,
        split: str = "train",
        crop_size: Optional[int] = 128,
        seed: int = 42,
        max_samples: Optional[int] = None,
        splits_path=None,
        manifest_path=None,
    ):
        """
        Args:
            split: 'train' (3,500 BOSSbase) or 'val' (500 BOSSbase).
            crop_size: Size of square crop for training (e.g. 128). None for full image (512).
            seed: Random seed for reproducible crop generation.
            max_samples: Optional limit on number of samples per epoch.
        """
        if split not in ("train", "val"):
            raise ValueError(f"RDHCheckerboardDataset only accepts 'train' or 'val', got '{split}'")

        self.split = split
        self.crop_size = crop_size
        self.cfg = get_config()

        split_file = splits_path or self.cfg.paths.splits_path
        manifest_file = manifest_path or self.cfg.paths.manifest_path
        splits_data, manifest_data, _ = load_split_manifest(
            split_file, manifest_file
        )

        self.filenames = splits_data[split]
        if max_samples is not None and max_samples < len(self.filenames):
            self.filenames = self.filenames[:max_samples]

        self.manifest_entries = [manifest_data[fn] for fn in self.filenames]
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.manifest_entries)

    def __getitem__(self, idx: int) -> dict:
        entry = self.manifest_entries[idx]
        file_path = entry["absolute_path"]
        img_arr, _ = load_grayscale_image(file_path, target_size=self.cfg.dataset.target_size)
        H, W = img_arr.shape

        # 1. Random crop for training, or center/full crop for validation
        if self.crop_size is not None and self.crop_size < min(H, W):
            cs = self.crop_size
            if self.split == "train":
                top = self.rng.randint(0, H - cs)
                left = self.rng.randint(0, W - cs)
            else:
                top = (H - cs) // 2
                left = (W - cs) // 2
            crop = img_arr[top:top+cs, left:left+cs]
        else:
            crop = img_arr
            cs = H

        # 2. Randomly select target set: Cross (0) or Dot (1)
        # Even-indexed epochs/samples can predict Cross, odd can predict Dot
        predict_cross = (idx % 2 == 0) if self.split == "val" else (self.rng.random() < 0.5)
        target_name = "cross" if predict_cross else "dot"

        # 3. Checkerboard masks
        r = np.arange(cs)[:, None]
        c = np.arange(cs)[None, :]
        cross_m = ((r + c) % 2 == 0)
        dot_m = ~cross_m
        target_m = cross_m if predict_cross else dot_m
        context_m = ~target_m

        # Interior mask (exclude 1-pixel border for stable loss computation)
        eval_m = target_m.copy()
        eval_m[0, :] = False
        eval_m[-1, :] = False
        eval_m[:, 0] = False
        eval_m[:, -1] = False

        # 4. Compute Rhombus prior on this crop
        rp_pred, _ = predict_rhombus(crop, target_set=target_name, exclude_border=False)

        # 5. Build 3-channel input tensor:
        # Channel 0: Normalized context image (target pixels strictly 0.0)
        norm_img = crop.astype(np.float32) / 255.0
        ch0 = norm_img * context_m.astype(np.float32)
        # Channel 1: Context indicator mask (1.0 for context, 0.0 for target)
        ch1 = context_m.astype(np.float32)
        # Channel 2: Rhombus prior
        ch2 = rp_pred.astype(np.float32) / 255.0

        input_tensor = torch.from_numpy(np.stack([ch0, ch1, ch2], axis=0))  # (3, cs, cs)
        target_tensor = torch.from_numpy(crop.astype(np.float32)).unsqueeze(0)  # (1, cs, cs) in [0, 255]
        eval_mask_tensor = torch.from_numpy(eval_m).unsqueeze(0)  # (1, cs, cs) bool

        return {
            "input": input_tensor,
            "target": target_tensor,
            "eval_mask": eval_mask_tensor,
            "filename": entry["filename"],
        }
