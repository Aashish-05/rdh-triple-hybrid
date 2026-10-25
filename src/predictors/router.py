"""
Hybrid Complexity Router for RDH Triple-Hybrid Predictor Framework.

Routing Logic:
1. Local Context Complexity:
   For target pixel x(i, j) on the checkerboard grid, the local complexity C(i, j)
   is computed strictly from its 4 orthogonal context pixels (top, bottom, left, right):
     Delta_grad = (|top - bottom| + |left - right|) / 2.0
     Delta_std  = std({top, bottom, left, right})
     C(i, j)    = (Delta_grad + Delta_std) / 2.0
   Since all 4 orthogonal neighbors strictly belong to the context set S_context,
   target pixel x(i, j) is completely excluded (Zero Target Leakage).
   Crucially, this ensures the decoder can reconstruct the EXACT same routing map
   without knowledge of the original cover image!

2. 3-Way Partitioning with Tunable Thresholds (T1, T2):
   - C(i, j) < T1:
     Route to Rhombus Predictor (RP) [Code 0]
     (Smooth / low-complexity region: fast, high zero-error probability).
   - T1 <= C(i, j) < T2:
     Route to CNN Predictor (CNNP) [Code 1]
     (Medium complexity / structural edges: convolutional receptive field).
   - C(i, j) >= T2:
     Route to Transformer Predictor (TFP) [Code 2]
     (High complexity / dense textures: window-restricted self-attention).

3. Dual Hybrid Mode (T2 = inf or T2 <= T1):
   - C(i, j) < T1: RP [Code 0]
   - C(i, j) >= T1: CNNP [Code 1]
"""
from typing import Tuple, Dict, Optional, Union
import numpy as np
import torch

from src.data.partition import get_checkerboard_masks
from src.predictors.rhombus import predict_rhombus, compute_prediction_errors
from src.predictors.cnn import CNNPixelPredictor, predict_cnn_image
from src.predictors.transformer import WindowTransformerPredictor, predict_transformer_image


def compute_local_rhombus_complexity(
    image: np.ndarray,
    exclude_border: bool = True,
) -> np.ndarray:
    """
    Compute local rhombus neighborhood complexity for every pixel strictly from context neighbors.

    Args:
        image: 2D uint8 numpy array (H, W).
        exclude_border: If True, borders are kept as 0.0.

    Returns:
        complexity_map: np.ndarray of shape (H, W), dtype float64.
    """
    arr = image.astype(np.float64)
    H, W = arr.shape
    c_map = np.zeros((H, W), dtype=np.float64)

    if H <= 2 or W <= 2:
        return c_map

    top = arr[0:H-2, 1:W-1]
    bottom = arr[2:H, 1:W-1]
    left = arr[1:H-1, 0:W-2]
    right = arr[1:H-1, 2:W]

    # Orthogonal gradient energy
    grad = 0.5 * (np.abs(top - bottom) + np.abs(left - right))
    # 4-neighbor standard deviation
    stack = np.stack([top, bottom, left, right], axis=0)
    std = np.std(stack, axis=0)

    # Combined local texture score
    c_map[1:H-1, 1:W-1] = 0.5 * (grad + std)

    if not exclude_border:
        # Fallback for boundary pixels
        for i in range(H):
            for j in range(W):
                if 1 <= i < H - 1 and 1 <= j < W - 1:
                    continue
                nbrs = []
                if i > 0: nbrs.append(arr[i - 1, j])
                if i < H - 1: nbrs.append(arr[i + 1, j])
                if j > 0: nbrs.append(arr[i, j - 1])
                if j < W - 1: nbrs.append(arr[i, j + 1])
                if len(nbrs) >= 2:
                    c_map[i, j] = float(np.std(nbrs))

    return c_map


def compute_routing_map(
    complexity_map: np.ndarray,
    T1: float,
    T2: float,
    eval_mask: np.ndarray,
) -> np.ndarray:
    """
    Generate predictor routing assignment map:
    - Code 0: RP  (C < T1)
    - Code 1: CNN (T1 <= C < T2)
    - Code 2: TFP (C >= T2)
    - Code -1: Un-evaluated pixels (e.g. outer boundary)

    Args:
        complexity_map: 2D float64 array of local complexity.
        T1: Lower threshold.
        T2: Upper threshold (T2 > T1).
        eval_mask: Boolean mask of valid evaluated pixels.

    Returns:
        routing_map: 2D int8 array of shape (H, W).
    """
    H, W = complexity_map.shape
    routing = np.full((H, W), fill_value=-1, dtype=np.int8)

    # Mask for RP: evaluated AND complexity < T1
    rp_mask = eval_mask & (complexity_map < T1)
    # Mask for CNN: evaluated AND T1 <= complexity < T2
    cnn_mask = eval_mask & (complexity_map >= T1) & (complexity_map < T2)
    # Mask for TFP: evaluated AND complexity >= T2
    tfp_mask = eval_mask & (complexity_map >= T2)

    routing[rp_mask] = 0
    routing[cnn_mask] = 1
    routing[tfp_mask] = 2

    return routing


def predict_triple_hybrid(
    image: np.ndarray,
    cnn_model: Optional[CNNPixelPredictor],
    tfp_model: Optional[WindowTransformerPredictor],
    T1: float,
    T2: float,
    target_set: str = "both",
    exclude_border: bool = True,
    device: str = "cuda",
    candidate_predictions: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Execute Hybrid Prediction across the image using the Complexity Router.

    Args:
        image: 2D uint8 numpy array (H, W).
        cnn_model: Trained CNN model (optional if candidate_predictions provided).
        tfp_model: Trained Transformer model (optional if candidate_predictions provided).
        T1: Lower threshold (RP / CNN).
        T2: Upper threshold (CNN / TFP). If T2 >= 1e8, operates in Dual Hybrid (RP+CNN) mode.
        target_set: 'cross', 'dot', or 'both'.
        exclude_border: Exclude 1-pixel border.
        device: 'cuda' or 'cpu'.
        candidate_predictions: Optional pre-computed dict containing 'rp', 'cnn', 'tfp' arrays
                              to accelerate validation grid search without redundant forward passes.

    Returns:
        (pred_image, eval_mask, routing_map, stats):
            - pred_image: Combined 2D uint8 array of predicted pixels.
            - eval_mask:  Boolean 2D array of evaluated locations.
            - routing_map: 2D int8 array with predictor codes (0: RP, 1: CNN, 2: TFP, -1: none).
            - stats: Dictionary with pixel counts and percentages assigned to each predictor.
    """
    H, W = image.shape
    c_map = compute_local_rhombus_complexity(image, exclude_border=exclude_border)

    # Get baseline RP prediction and eval mask
    if candidate_predictions and "rp" in candidate_predictions:
        rp_pred = candidate_predictions["rp"]
        eval_mask = candidate_predictions["eval_mask"]
    else:
        rp_pred, eval_mask = predict_rhombus(image, target_set=target_set, exclude_border=exclude_border)

    # Compute routing map
    routing_map = compute_routing_map(c_map, T1, T2, eval_mask)

    rp_active = (routing_map == 0)
    cnn_active = (routing_map == 1)
    tfp_active = (routing_map == 2)

    total_eval = int(eval_mask.sum())
    count_rp = int(rp_active.sum())
    count_cnn = int(cnn_active.sum())
    count_tfp = int(tfp_active.sum())

    # Build predictions
    pred_image = np.zeros((H, W), dtype=np.uint8)

    # 1. Assign RP predictions
    pred_image[rp_active] = rp_pred[rp_active]

    # 2. Assign CNN predictions
    if count_cnn > 0:
        if candidate_predictions and "cnn" in candidate_predictions:
            cnn_pred = candidate_predictions["cnn"]
        else:
            if cnn_model is None:
                raise ValueError("cnn_model must be provided when pixels are routed to CNN")
            cnn_pred, _ = predict_cnn_image(cnn_model, image, target_set=target_set, exclude_border=exclude_border, device=device)
        pred_image[cnn_active] = cnn_pred[cnn_active]

    # 3. Assign Transformer predictions
    if count_tfp > 0:
        if candidate_predictions and "tfp" in candidate_predictions:
            tfp_pred = candidate_predictions["tfp"]
        else:
            if tfp_model is None:
                raise ValueError("tfp_model must be provided when pixels are routed to Transformer")
            tfp_pred, _ = predict_transformer_image(tfp_model, image, target_set=target_set, exclude_border=exclude_border, device=device)
        pred_image[tfp_active] = tfp_pred[tfp_active]

    stats = {
        "total_evaluated_pixels": total_eval,
        "rp_pixel_count": count_rp,
        "cnn_pixel_count": count_cnn,
        "tfp_pixel_count": count_tfp,
        "rp_percentage": round((count_rp / max(1, total_eval)) * 100.0, 2),
        "cnn_percentage": round((count_cnn / max(1, total_eval)) * 100.0, 2),
        "tfp_percentage": round((count_tfp / max(1, total_eval)) * 100.0, 2),
        "T1": float(T1),
        "T2": float(T2),
    }

    return pred_image, eval_mask, routing_map, stats
