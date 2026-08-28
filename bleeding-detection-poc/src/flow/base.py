from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class DenseFlowEstimator(ABC):
    @abstractmethod
    def estimate(self, first_roi: np.ndarray, second_roi: np.ndarray) -> np.ndarray:
        """Return H x W x 2 float32 pixel flow in original ROI coordinates."""
        raise NotImplementedError
