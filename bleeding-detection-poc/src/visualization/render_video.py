from __future__ import annotations

import cv2
import numpy as np

from src.preprocessing.bbox_crop import BoundingBox

CVAT_COLORS = {
    "bleeding-area": (245, 61, 61),
    "blood-origin": (61, 245, 61),
    "bleed-origin": (61, 245, 61),
}


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


def mask_to_bgr(mask: np.ndarray) -> np.ndarray:
    return cv2.cvtColor((mask.astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR)


def draw_bbox(frame: np.ndarray, bbox: BoundingBox) -> np.ndarray:
    result = frame.copy()
    cv2.rectangle(result, (bbox.x, bbox.y), (bbox.x + bbox.width, bbox.y + bbox.height), (0, 255, 255), 2)
    return result


def draw_cvat_context(frame: np.ndarray, bleeding_box: BoundingBox | None, origin_shapes: list) -> np.ndarray:
    """Draw CVAT bleeding boundaries and interpolated origin tracks."""
    result = frame.copy()
    if bleeding_box is not None:
        cv2.rectangle(
            result,
            (bleeding_box.x, bleeding_box.y),
            (bleeding_box.x + bleeding_box.width, bleeding_box.y + bleeding_box.height),
            CVAT_COLORS["bleeding-area"],
            3,
            cv2.LINE_AA,
        )
        cv2.putText(result, "bleeding-area", (bleeding_box.x, max(24, bleeding_box.y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, CVAT_COLORS["bleeding-area"], 2, cv2.LINE_AA)
    for shape in origin_shapes:
        points = np.rint(shape.points).astype(np.int32)
        color = CVAT_COLORS.get(shape.label, (0, 255, 255))
        if shape.shape_type == "polyline":
            cv2.polylines(result, [points], False, color, 3, cv2.LINE_AA)
            label_point = tuple(points[0])
        else:
            label_point = tuple(points[0])
            cv2.drawMarker(result, label_point, color, cv2.MARKER_CROSS, 18, 3, cv2.LINE_AA)
        cv2.putText(result, shape.label, (label_point[0] + (12 if shape.shape_type == "points" else 0), max(24, label_point[1] - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return result


def full_frame_overlay(frame: np.ndarray, flow_bgr: np.ndarray, mask: np.ndarray, bbox: BoundingBox, alpha: float) -> np.ndarray:
    result = frame.copy()
    roi = result[bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width]
    roi[mask] = cv2.addWeighted(roi, 1.0 - alpha, flow_bgr, alpha, 0)[mask]
    return draw_bbox(result, bbox)


def render_dashboard(
    frame: np.ndarray,
    roi: np.ndarray,
    flow_bgr: np.ndarray,
    magnitude_bgr: np.ndarray,
    instrument_mask_bgr: np.ndarray,
    flow_suppressed_bgr: np.ndarray,
    mask: np.ndarray,
    bbox: BoundingBox,
) -> np.ndarray:
    top = draw_bbox(frame, bbox)
    panel_height, panel_width = top.shape[:2]
    panel_height = max(1, panel_height // 3)
    panel_width = panel_width // 3
    mask_bgr = mask_to_bgr(mask)
    panels = [roi, flow_bgr, magnitude_bgr, instrument_mask_bgr, flow_suppressed_bgr, mask_bgr]
    labels = [
        "Cropped ROI",
        "NeuFlow v2 (raw)",
        "Raw magnitude",
        "Instrument mask",
        "Flow after suppression",
        "Candidate mask",
    ]
    rendered = []
    for panel, label in zip(panels, labels):
        panel = cv2.resize(panel, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
        cv2.putText(panel, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        rendered.append(panel)
    bottom = np.vstack([np.hstack(rendered[:3]), np.hstack(rendered[3:])])
    top = cv2.resize(top, (bottom.shape[1], top.shape[0] * bottom.shape[1] // top.shape[1]), interpolation=cv2.INTER_AREA)
    return np.vstack([top, bottom])
