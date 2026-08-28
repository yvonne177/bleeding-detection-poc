from __future__ import annotations

from collections import deque

import cv2
import numpy as np


def instrument_mask_from_magnitude(magnitude: np.ndarray, motion_threshold: float, dilation_kernel_size: int) -> np.ndarray:
    """Heuristic instrument mask: fast optical-flow motion, dilated to cover tool edges.

    This does not distinguish instruments from other fast motion; it assumes the
    dominant fast-moving regions in the ROI are instruments, matching the observed
    NeuFlow outlines.
    """
    mask = (magnitude >= motion_threshold).astype(np.uint8)
    if dilation_kernel_size > 1:
        kernel = np.ones((dilation_kernel_size, dilation_kernel_size), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel)
    return mask.astype(bool)


class InstrumentMaskTracker:
    """Keeps a short rolling union of instrument masks so a tool leaving a pixel does not instantly clear it."""

    def __init__(self, persistence_frames: int) -> None:
        self.history: deque[np.ndarray] = deque(maxlen=max(1, persistence_frames))

    def update(self, current_mask: np.ndarray) -> np.ndarray:
        self.history.append(current_mask)
        combined = current_mask
        for previous in self.history:
            combined = combined | previous
        return combined


def suppress_instrument_flow(flow: np.ndarray, instrument_mask: np.ndarray) -> np.ndarray:
    """Zero flow vectors at instrument-mask pixels; leaves all other ROI pixels untouched."""
    suppressed = flow.copy()
    suppressed[instrument_mask] = 0.0
    return suppressed
