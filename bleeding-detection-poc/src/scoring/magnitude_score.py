from __future__ import annotations

import numpy as np


def flow_magnitude(flow: np.ndarray) -> np.ndarray:
    """Return pixel displacement magnitude for H x W x 2 (dx, dy) flow."""
    return np.linalg.norm(flow, axis=2)


def candidate_mask(magnitude: np.ndarray, threshold: float) -> np.ndarray:
    return magnitude > threshold
