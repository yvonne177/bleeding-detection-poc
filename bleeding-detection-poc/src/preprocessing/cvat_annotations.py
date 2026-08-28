from __future__ import annotations

import json
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.preprocessing.bbox_crop import BoundingBox


@dataclass(frozen=True)
class CvatShape:
    label: str
    frame: int
    shape_type: str
    points: np.ndarray
    outside: bool


@dataclass(frozen=True)
class CvatTrack:
    label: str
    shapes: list[CvatShape]


class CvatAnnotations:
    """Read CVAT JSON tracks used as manual ROI/origin context, not ground truth."""

    def __init__(self, annotation_path: Path, task_path: Path | None = None) -> None:
        document = json.loads(annotation_path.read_text(encoding="utf-8"))
        if not isinstance(document, list) or len(document) != 1:
            raise ValueError("Expected a single CVAT JSON annotation document.")
        if task_path is not None:
            task = json.loads(task_path.read_text(encoding="utf-8"))
            label_names = {label["name"] for label in task.get("labels", [])}
            required = {"bleeding-area", "blood-origin"}
            missing = required - label_names
            if missing:
                raise ValueError(f"CVAT task is missing required labels: {sorted(missing)}")

        tracks = document[0].get("tracks", [])
        self.bleeding_tracks = self._tracks_for_labels(tracks, {"bleeding-area"}, {"rectangle"})
        self.origin_tracks = self._tracks_for_labels(
            tracks, {"blood-origin", "bleed-origin"}, {"polyline", "points"}
        )
        if not self.bleeding_tracks:
            raise ValueError("CVAT export contains no bleeding-area rectangle annotations.")

    @staticmethod
    def _tracks_for_labels(tracks: list[dict], labels: set[str], allowed_types: set[str]) -> list[CvatTrack]:
        parsed_tracks = []
        for track in tracks:
            label = track.get("label")
            if label not in labels:
                continue
            shapes = []
            for shape in track.get("shapes", []):
                if shape["type"] in allowed_types:
                    shapes.append(
                        CvatShape(
                            label=label,
                            frame=int(shape["frame"]),
                            shape_type=shape["type"],
                            points=np.asarray(shape["points"], dtype=np.float32).reshape(-1, 2),
                            outside=bool(shape.get("outside", False)),
                        )
                    )
            if shapes:
                parsed_tracks.append(CvatTrack(label=label, shapes=sorted(shapes, key=lambda shape: shape.frame)))
        return parsed_tracks

    @staticmethod
    def _interpolated_shape(shapes: list[CvatShape], frame_index: int) -> CvatShape | None:
        index = bisect_right([shape.frame for shape in shapes], frame_index) - 1
        if index < 0:
            return None
        current = shapes[index]
        if current.outside or index == len(shapes) - 1:
            return None if current.outside else current
        following = shapes[index + 1]
        if current.shape_type != following.shape_type or current.points.shape != following.points.shape:
            return current
        if following.outside:
            return current
        fraction = (frame_index - current.frame) / (following.frame - current.frame)
        return CvatShape(
            label=current.label,
            frame=frame_index,
            shape_type=current.shape_type,
            points=current.points + fraction * (following.points - current.points),
            outside=False,
        )

    def bleeding_box_at(self, frame_index: int) -> BoundingBox | None:
        shapes = [self._interpolated_shape(track.shapes, frame_index) for track in self.bleeding_tracks]
        shapes = [shape for shape in shapes if shape is not None]
        if not shapes:
            return None
        points = np.vstack([shape.points for shape in shapes])
        x0, y0 = points.min(axis=0)
        x1, y1 = points.max(axis=0)
        return BoundingBox(int(np.floor(x0)), int(np.floor(y0)), int(np.ceil(x1 - x0)), int(np.ceil(y1 - y0)))

    def fixed_roi_for_range(self, start_frame: int, end_frame: int, frame_width: int, frame_height: int, padding: int = 0) -> BoundingBox:
        boxes = [self.bleeding_box_at(frame) for frame in range(start_frame, end_frame)]
        boxes = [box for box in boxes if box is not None]
        if not boxes:
            raise ValueError("No active bleeding-area rectangle annotations in the selected frame range.")
        x0 = max(0, min(box.x for box in boxes) - padding)
        y0 = max(0, min(box.y for box in boxes) - padding)
        x1 = min(frame_width, max(box.x + box.width for box in boxes) + padding)
        y1 = min(frame_height, max(box.y + box.height for box in boxes) + padding)
        return BoundingBox(x0, y0, x1 - x0, y1 - y0)

    def annotated_frame_range(self) -> tuple[int, int]:
        """Return the inclusive-to-exclusive source range covered by manual CVAT labels."""
        shapes = [
            shape
            for track in self.bleeding_tracks + self.origin_tracks
            for shape in track.shapes
            if not shape.outside
        ]
        if not shapes:
            raise ValueError("CVAT export contains no visible bleeding-area or blood-origin annotations.")
        return min(shape.frame for shape in shapes), max(shape.frame for shape in shapes) + 1

    def bleeding_mask_for_roi(self, frame_index: int, roi: BoundingBox) -> np.ndarray | None:
        box = self.bleeding_box_at(frame_index)
        if box is None:
            return None
        mask = np.zeros((roi.height, roi.width), dtype=bool)
        x0 = max(0, box.x - roi.x)
        y0 = max(0, box.y - roi.y)
        x1 = min(roi.width, box.x + box.width - roi.x)
        y1 = min(roi.height, box.y + box.height - roi.y)
        if x0 < x1 and y0 < y1:
            mask[y0:y1, x0:x1] = True
        return mask

    def origin_shapes_at(self, frame_index: int) -> list[CvatShape]:
        return [
            shape
            for track in self.origin_tracks
            if (shape := self._interpolated_shape(track.shapes, frame_index)) is not None
        ]
