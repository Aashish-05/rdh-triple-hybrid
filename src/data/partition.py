"""
Checkerboard Cross/Dot Partitioning for RDH Triple-Hybrid Predictor Framework.

In two-pass Reversible Data Hiding (RDH):
1. The image grid I of size H x W is decomposed into two mutually exclusive,
   collectively exhaustive subsets:
     - Cross Set (S_C): Pixels (r, c) where (r + c) % 2 == 0
     - Dot Set   (S_D): Pixels (r, c) where (r + c) % 2 == 1
2. For any pixel in S_C, all four of its orthogonal nearest neighbors
   (top, bottom, left, right) strictly belong to S_D.
3. In Pass 1, Dot pixels remain unmodified and serve as the context to predict
   and embed secret payload into Cross pixels.
4. In Pass 2, the marked Cross pixels serve as the context to predict and embed
   payload into Dot pixels.

Note: Partitioning is the structural foundation for context isolation in 2-pass RDH.
It does not itself implement reversible data embedding or extraction.
"""
from typing import Tuple, Union
import numpy as np
import torch


def get_checkerboard_masks(
    height: int,
    width: int,
    device: Union[str, torch.device, None] = None,
    as_tensor: bool = False,
) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    """
    Generate boolean masks for Cross and Dot sets.

    Args:
        height: Image height H >= 1.
        width: Image width W >= 1.
        device: PyTorch device (used if as_tensor=True).
        as_tensor: If True, returns torch.BoolTensor; if False, returns np.ndarray (bool).

    Returns:
        (cross_mask, dot_mask):
            - cross_mask: True where (r + c) % 2 == 0.
            - dot_mask:   True where (r + c) % 2 == 1.
    """
    if height < 1 or width < 1:
        raise ValueError(f"Height and width must be >= 1, got ({height}, {width})")

    r = np.arange(height)[:, None]
    c = np.arange(width)[None, :]
    cross_mask = ((r + c) % 2 == 0)
    dot_mask = ~cross_mask

    if as_tensor:
        t_cross = torch.from_numpy(cross_mask).to(device=device)
        t_dot = torch.from_numpy(dot_mask).to(device=device)
        return t_cross, t_dot

    return cross_mask, dot_mask


def split_image_cross_dot(
    img: Union[np.ndarray, torch.Tensor]
) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor], Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]]:
    """
    Split an image into 1D arrays of Cross and Dot pixel values.

    Args:
        img: 2D array of shape (H, W) or 3D tensor/array of shape (1, H, W).

    Returns:
        (cross_values, dot_values, (cross_mask, dot_mask)):
            - cross_values: 1D array/tensor containing all pixels where (r + c) % 2 == 0.
            - dot_values:   1D array/tensor containing all pixels where (r + c) % 2 == 1.
            - masks: Tuple of 2D boolean masks (cross_mask, dot_mask).
    """
    is_torch = isinstance(img, torch.Tensor)

    if is_torch:
        if img.ndim == 3 and img.shape[0] == 1:
            squeeze_img = img.squeeze(0)
        elif img.ndim == 2:
            squeeze_img = img
        else:
            raise ValueError(f"Expected 2D (H, W) or 3D (1, H, W) tensor, got shape {img.shape}")

        h, w = squeeze_img.shape
        cross_m, dot_m = get_checkerboard_masks(h, w, device=img.device, as_tensor=True)
        cross_vals = squeeze_img[cross_m]
        dot_vals = squeeze_img[dot_m]
        return cross_vals, dot_vals, (cross_m, dot_m)
    else:
        arr = np.asarray(img)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim != 2:
            raise ValueError(f"Expected 2D (H, W) or 3D (1, H, W) array, got shape {arr.shape}")

        h, w = arr.shape
        cross_m, dot_m = get_checkerboard_masks(h, w, as_tensor=False)
        cross_vals = arr[cross_m]
        dot_vals = arr[dot_m]
        return cross_vals, dot_vals, (cross_m, dot_m)


def reconstruct_image_from_cross_dot(
    cross_values: Union[np.ndarray, torch.Tensor],
    dot_values: Union[np.ndarray, torch.Tensor],
    height: int,
    width: int,
    dtype=None,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Reconstruct the original 2D image from Cross and Dot pixel values.

    Args:
        cross_values: 1D array/tensor of Cross pixel values.
        dot_values:   1D array/tensor of Dot pixel values.
        height:       Original image height H.
        width:        Original image width W.
        dtype:        Target data type (defaults to cross_values dtype).

    Returns:
        Reconstructed 2D image of shape (H, W) exactly matching the source.
    """
    is_torch = isinstance(cross_values, torch.Tensor)

    if is_torch:
        cross_m, dot_m = get_checkerboard_masks(height, width, device=cross_values.device, as_tensor=True)
        out = torch.empty((height, width), dtype=dtype or cross_values.dtype, device=cross_values.device)
        out[cross_m] = cross_values
        out[dot_m] = dot_values
        return out
    else:
        cross_m, dot_m = get_checkerboard_masks(height, width, as_tensor=False)
        out = np.empty((height, width), dtype=dtype or cross_values.dtype)
        out[cross_m] = cross_values
        out[dot_m] = dot_values
        return out
