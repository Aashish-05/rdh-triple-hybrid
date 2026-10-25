"""
Grayscale Image Loader and PyTorch Dataset for RDH Triple-Hybrid Predictor Framework.

Guarantees:
1. Preserves original 8-bit pixel values intact for grayscale images (mode 'L').
2. Deterministic ITU-R BT.601 conversion for RGB images: Y = 0.299*R + 0.587*G + 0.114*B.
3. Strict 512x512 dimension enforcement without resizing already-512x512 images.
4. Returns uint8 numpy arrays [0, 255] and PyTorch float32/uint8 tensors.
"""
from pathlib import Path
from typing import Tuple, Optional, Union, List
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.config import get_config


def load_grayscale_image(
    file_path: Union[str, Path],
    target_size: Tuple[int, int] = (512, 512),
    allow_resize: bool = False,
) -> Tuple[np.ndarray, dict]:
    """
    Load an image as a 2D grayscale uint8 numpy array [0, 255].

    Args:
        file_path: Path to .pgm, .tiff, or other supported image format.
        target_size: Expected (width, height) tuple, default (512, 512).
        allow_resize: If False, raises ValueError when image dimensions do not match target_size.

    Returns:
        (image_array, metadata):
            - image_array: np.ndarray of shape (H, W), dtype uint8.
            - metadata: dict recording source format, original mode, dimensions, and whether conversion/resizing occurred.
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Image file does not exist: {p}")

    with Image.open(p) as img:
        orig_mode = img.mode
        orig_size = img.size  # (width, height)
        resized = False
        converted = False

        # Mode handling
        if orig_mode == "L":
            # Native 8-bit grayscale: read pixel array directly to prevent any alterations
            gray_img = img
        elif orig_mode in ("RGB", "RGBA"):
            # Convert RGB to Grayscale using ITU-R BT.601 deterministic formula:
            # Y = round(0.299*R + 0.587*G + 0.114*B)
            if orig_mode == "RGBA":
                # Drop alpha channel first
                img = img.convert("RGB")
            gray_img = img.convert("L")
            converted = True
        elif orig_mode == "1":
            # 1-bit bilevel
            gray_img = img.convert("L")
            converted = True
        elif orig_mode in ("I", "I;16"):
            # 16-bit grayscale -> scale to 8-bit [0, 255]
            arr = np.array(img, dtype=np.float32)
            arr = (arr / arr.max() * 255.0).clip(0, 255).astype(np.uint8)
            gray_img = Image.fromarray(arr, mode="L")
            converted = True
        else:
            gray_img = img.convert("L")
            converted = True

        # Size check and dimension enforcement
        if gray_img.size != target_size:
            if not allow_resize:
                raise ValueError(
                    f"Image {p.name} has dimensions {gray_img.size}, expected {target_size}. "
                    "allow_resize is set to False."
                )
            # Bilinear interpolation if resizing is explicitly permitted
            gray_img = gray_img.resize(target_size, resample=Image.Resampling.BILINEAR)
            resized = True

        arr = np.array(gray_img, dtype=np.uint8)

    meta = {
        "filename": p.name,
        "path": str(p),
        "orig_mode": orig_mode,
        "orig_size": orig_size,
        "target_size": target_size,
        "converted_to_gray": converted,
        "resized": resized,
        "dtype": str(arr.dtype),
        "shape": arr.shape,
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }
    return arr, meta


class RDHImageDataset(Dataset):
    """
    PyTorch Dataset providing images from the approved train, val, or test splits.
    """
    def __init__(
        self,
        split: str = "train",
        transform=None,
        as_float: bool = True,
        normalize_01: bool = True,
    ):
        """
        Args:
            split: One of 'train', 'val', or 'test'.
            transform: Optional torchvision-style transform.
            as_float: If True, returns torch.FloatTensor; if False, returns torch.uint8.
            normalize_01: If True and as_float=True, scales pixel values from [0, 255] to [0.0, 1.0].
        """
        if split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split '{split}'. Must be 'train', 'val', or 'test'.")

        self.split = split
        self.transform = transform
        self.as_float = as_float
        self.normalize_01 = normalize_01
        self.cfg = get_config()

        # Load split definitions and manifest
        with open(self.cfg.paths.splits_path, "r") as f:
            splits_data = json.load(f)
        with open(self.cfg.paths.manifest_path, "r") as f:
            manifest_data = json.load(f)

        self.filenames: List[str] = splits_data[split]
        self.samples = [manifest_data[fn] for fn in self.filenames]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        entry = self.samples[idx]
        file_path = entry["absolute_path"]
        arr, _ = load_grayscale_image(file_path, target_size=self.cfg.dataset.target_size)

        if self.as_float:
            tensor = torch.from_numpy(arr).float()
            if self.normalize_01:
                tensor = tensor / 255.0
        else:
            tensor = torch.from_numpy(arr)

        # Add channel dimension: (1, H, W)
        tensor = tensor.unsqueeze(0)

        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor, entry["filename"]
