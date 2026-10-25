"""
Rhombus Predictor (RP) Baseline for RDH Triple-Hybrid Predictor Framework.

Mathematical Foundation & Baseline Reference:
Based on Zhao et al. (CSSP 2026 / 2025, DOI: 10.1007/s00034-025-03332-1) and
classic checkerboard RDH (Sachnev et al., IEEE T-CSVT 2009):

1. For an interior target pixel x(i, j), the 4 orthogonal context pixels are:
     - Top:    v1 = x(i - 1, j)
     - Bottom: v2 = x(i + 1, j)
     - Left:   v3 = x(i, j - 1)
     - Right:  v4 = x(i, j + 1)

2. Due to the checkerboard partition, if x(i, j) in S_C (Cross), then v1, v2, v3, v4 in S_D (Dot).
   This guarantees that no target pixel participates in its own prediction (zero leakage).

3. Exact Integer Formula:
     x_hat(i, j) = floor( (v1 + v2 + v3 + v4 + 2) / 4 )
   This represents exact half-up nearest integer rounding: round((v1 + v2 + v3 + v4) / 4.0).

4. Border Handling:
   In standard RDH (Zhao et al. & Sachnev et al.), the outer 1-pixel border of the image
   (rows 0, H-1; columns 0, W-1) is excluded from data embedding and kept as fixed context
   and auxiliary book-keeping storage.
   When exclude_border=False, border pixels are predicted using their available 2 (corners)
   or 3 (edges) orthogonal neighbors with symmetric nearest integer rounding:
     floor( (sum(v) + floor(m / 2)) / m )
"""
from typing import Tuple, Optional, Union
import numpy as np
import torch


def predict_rhombus_single(neighbors: list) -> int:
    """
    Predict a single pixel value given its orthogonal neighbors using integer rounding.

    Args:
        neighbors: List of 2, 3, or 4 integer pixel values in [0, 255].

    Returns:
        Integer prediction in [0, 255].
    """
    m = len(neighbors)
    if m == 0:
        raise ValueError("Neighbors list cannot be empty")
    s = sum(neighbors)
    pred = (s + m // 2) // m
    return max(0, min(255, pred))


def predict_rhombus(
    image: np.ndarray,
    target_set: str = "cross",
    exclude_border: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Rhombus Predictor (RP) values across the entire image for a specified subset.

    Args:
        image: 2D numpy array of shape (H, W), dtype uint8 or int32.
        target_set: 'cross' (predict pixels where (r+c)%2 == 0),
                    'dot'   (predict pixels where (r+c)%2 == 1),
                    or 'both' (predict both sets independently from their respective contexts).
        exclude_border: If True, excludes the outer 1-pixel boundary from prediction
                        (standard RDH protocol to match baseline paper).
                        If False, borders are predicted using available 2/3 neighbors.

    Returns:
        (pred_image, eval_mask):
            - pred_image: np.ndarray of shape (H, W), dtype uint8, containing predictions at evaluated locations.
            - eval_mask:  Boolean np.ndarray of shape (H, W), indicating which pixels were predicted.
    """
    arr = np.asarray(image, dtype=np.int32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D image array (H, W), got shape {arr.shape}")

    H, W = arr.shape
    pred_image = np.zeros((H, W), dtype=np.uint8)
    eval_mask = np.zeros((H, W), dtype=bool)

    if H < 1 or W < 1:
        return pred_image, eval_mask

    # Checkerboard masks
    r = np.arange(H)[:, None]
    c = np.arange(W)[None, :]
    cross_mask = ((r + c) % 2 == 0)
    dot_mask = ~cross_mask

    if target_set.lower() == "cross":
        target_mask = cross_mask
    elif target_set.lower() == "dot":
        target_mask = dot_mask
    elif target_set.lower() == "both":
        target_mask = np.ones((H, W), dtype=bool)
    else:
        raise ValueError(f"Invalid target_set '{target_set}'. Must be 'cross', 'dot', or 'both'.")

    # Define interior mask
    interior_mask = np.zeros((H, W), dtype=bool)
    if H > 2 and W > 2:
        interior_mask[1:H-1, 1:W-1] = True

    # 1. Vectorized computation for interior pixels (all 4 neighbors present)
    if H > 2 and W > 2:
        interior_targets = target_mask[1:H-1, 1:W-1]
        top = arr[0:H-2, 1:W-1]
        bottom = arr[2:H, 1:W-1]
        left = arr[1:H-1, 0:W-2]
        right = arr[1:H-1, 2:W]

        # Exact formula: (top + bottom + left + right + 2) // 4
        interior_pred = (top + bottom + left + right + 2) // 4
        interior_pred = np.clip(interior_pred, 0, 255).astype(np.uint8)

        # Place into pred_image
        pred_slice = pred_image[1:H-1, 1:W-1]
        pred_slice[interior_targets] = interior_pred[interior_targets]
        eval_mask[1:H-1, 1:W-1] = interior_targets

    # 2. Border pixel computation (if exclude_border is False)
    if not exclude_border:
        for i in range(H):
            for j in range(W):
                # Skip interior pixels already computed
                if 1 <= i < H - 1 and 1 <= j < W - 1:
                    continue
                if not target_mask[i, j]:
                    continue

                nbrs = []
                if i > 0:
                    nbrs.append(int(arr[i - 1, j]))
                if i < H - 1:
                    nbrs.append(int(arr[i + 1, j]))
                if j > 0:
                    nbrs.append(int(arr[i, j - 1]))
                if j < W - 1:
                    nbrs.append(int(arr[i, j + 1]))

                if nbrs:
                    pred_image[i, j] = predict_rhombus_single(nbrs)
                    eval_mask[i, j] = True

    return pred_image, eval_mask


def compute_prediction_errors(
    image: np.ndarray,
    pred_image: np.ndarray,
    eval_mask: np.ndarray,
) -> np.ndarray:
    """
    Compute signed prediction errors: e = x - x_hat for all evaluated pixels.

    Args:
        image: Original image array (H, W).
        pred_image: Predicted image array (H, W).
        eval_mask: Boolean mask indicating evaluated pixels.

    Returns:
        1D int32 array containing prediction errors (x - x_hat).
    """
    orig = np.asarray(image, dtype=np.int32)
    pred = np.asarray(pred_image, dtype=np.int32)
    errors = orig[eval_mask] - pred[eval_mask]
    return errors
