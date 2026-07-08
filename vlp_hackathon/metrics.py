from __future__ import annotations
import numpy as np


def euclidean_errors_cm(pred_xy: np.ndarray, true_xy: np.ndarray) -> np.ndarray:
    pred_xy = np.asarray(pred_xy, dtype=np.float32)
    true_xy = np.asarray(true_xy, dtype=np.float32)
    if pred_xy.shape != true_xy.shape or pred_xy.ndim != 2 or pred_xy.shape[1] != 2:
        raise ValueError(f"Expected matching (N,2) arrays, got {pred_xy.shape} and {true_xy.shape}")
    return np.linalg.norm(pred_xy - true_xy, axis=1)
