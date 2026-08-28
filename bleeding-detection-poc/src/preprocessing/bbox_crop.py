from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    @classmethod
    def from_config(cls, config: dict) -> "BoundingBox":
        return cls(**{key: int(config[key]) for key in ("x", "y", "width", "height")})

    def validate_for(self, frame: np.ndarray) -> None:
        frame_height, frame_width = frame.shape[:2]
        if self.width <= 0 or self.height <= 0:
            raise ValueError("ROI width and height must be positive.")
        if self.x < 0 or self.y < 0 or self.x + self.width > frame_width or self.y + self.height > frame_height:
            raise ValueError(
                f"ROI {self} is outside source frame dimensions {frame_width}x{frame_height}."
            )

    def crop(self, frame: np.ndarray) -> np.ndarray:
        self.validate_for(frame)
        return frame[self.y : self.y + self.height, self.x : self.x + self.width].copy()
