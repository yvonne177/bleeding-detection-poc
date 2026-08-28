from __future__ import annotations

import cv2
import numpy as np

from src.preprocessing.bbox_crop import BoundingBox


def flow_to_hsv_bgr(flow: np.ndarray, magnitude_clip: float) -> np.ndarray:
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    hsv = np.zeros((*magnitude.shape, 3), dtype=np.uint8)
    hsv[..., 0] = (angle / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(magnitude / magnitude_clip * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def magnitude_to_bgr(magnitude: np.ndarray, magnitude_clip: float) -> np.ndarray:
    image = np.clip(magnitude / magnitude_clip * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(image, cv2.COLORMAP_TURBO)


def draw_bbox(frame: np.ndarray, bbox: BoundingBox) -> np.ndarray:
    result = frame.copy()
    cv2.rectangle(result, (bbox.x, bbox.y), (bbox.x + bbox.width, bbox.y + bbox.height), (0, 255, 255), 2)
    return result


def full_frame_overlay(frame: np.ndarray, flow_bgr: np.ndarray, mask: np.ndarray, bbox: BoundingBox, alpha: float) -> np.ndarray:
    result = frame.copy()
    roi = result[bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width]
    roi[mask] = cv2.addWeighted(roi, 1.0 - alpha, flow_bgr, alpha, 0)[mask]
    return draw_bbox(result, bbox)


def render_dashboard(frame: np.ndarray, roi: np.ndarray, flow_bgr: np.ndarray, magnitude_bgr: np.ndarray, mask: np.ndarray, bbox: BoundingBox) -> np.ndarray:
    top = draw_bbox(frame, bbox)
    panel_height, panel_width = top.shape[:2]
    panel_height = max(1, panel_height // 3)
    panel_width = panel_width // 2
    mask_bgr = cv2.cvtColor((mask.astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR)
    panels = [roi, flow_bgr, magnitude_bgr, mask_bgr]
    labels = ["Cropped ROI", "NeuFlow v2", "Magnitude", "Candidate mask"]
    rendered = []
    for panel, label in zip(panels, labels):
        panel = cv2.resize(panel, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
        cv2.putText(panel, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        rendered.append(panel)
    bottom = np.vstack([np.hstack(rendered[:2]), np.hstack(rendered[2:])])
    return np.vstack([top, bottom])
