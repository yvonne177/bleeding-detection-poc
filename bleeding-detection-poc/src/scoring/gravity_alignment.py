from __future__ import annotations

import numpy as np


def gravity_alignment(flow: np.ndarray, gravity_x: float, gravity_y: float) -> np.ndarray:
    """Return cosine alignment with gravity; zero-magnitude flow has zero alignment."""
    gravity = np.array([gravity_x, gravity_y], dtype=np.float32)
    gravity_norm = np.linalg.norm(gravity)
    if gravity_norm == 0:
        raise ValueError("Gravity vector must not be zero.")
    magnitudes = np.linalg.norm(flow, axis=2)
    alignment = np.zeros_like(magnitudes, dtype=np.float32)
    valid = magnitudes > 0
    alignment[valid] = np.sum(flow[valid] * gravity, axis=1) / (magnitudes[valid] * gravity_norm)
    return alignment
