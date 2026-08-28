from __future__ import annotations

import cv2
import numpy as np


def blood_color_mask(
    roi_bgr: np.ndarray,
    hue_margin: float,
    minimum_saturation: float,
    maximum_value: float,
    opening_kernel_size: int,
) -> np.ndarray:
    """Select dark red pixels: red hue (wrapping 0/180 in OpenCV HSV), saturated, and dark.

    Perfused tissue is red but bright; pooled or running blood is red and dark, so the
    value ceiling is what separates blood from the tissue it flows over. Metal and
    plastic instruments fail the hue and saturation tests regardless of how fast they move.
    """
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[..., 0].astype(np.float32)
    saturation = hsv[..., 1].astype(np.float32)
    value = hsv[..., 2].astype(np.float32)
    red_hue = (hue <= hue_margin) | (hue >= 180.0 - hue_margin)
    mask = red_hue & (saturation >= minimum_saturation) & (value <= maximum_value)
    return _open(mask, opening_kernel_size)


def darkening_mask(
    previous_roi_bgr: np.ndarray | None,
    current_roi_bgr: np.ndarray,
    minimum_drop: float,
    blur_kernel_size: int,
) -> np.ndarray:
    """Select pixels that got darker than the previous frame by at least ``minimum_drop``.

    Blood spreading over tissue darkens the pixels it covers; specular tool highlights
    and lighting flicker are suppressed by the blur before differencing.
    """
    current_value = _blurred_value(current_roi_bgr, blur_kernel_size)
    if previous_roi_bgr is None:
        return np.zeros(current_value.shape, dtype=bool)
    previous_value = _blurred_value(previous_roi_bgr, blur_kernel_size)
    return (previous_value - current_value) >= minimum_drop


def _blurred_value(roi_bgr: np.ndarray, blur_kernel_size: int) -> np.ndarray:
    value = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)[..., 2].astype(np.float32)
    if blur_kernel_size > 1:
        kernel = blur_kernel_size | 1
        value = cv2.GaussianBlur(value, (kernel, kernel), 0)
    return value


def _open(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return mask
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    return opened.astype(bool)
