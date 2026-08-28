"""Render CVAT video-track annotations onto their source video."""

import argparse
import json
from bisect import bisect_right
from pathlib import Path

import cv2


COLORS = {
    "bleeding-area": (245, 61, 61),
    "blood-origin": (61, 245, 61),
    "bleed-origin": (61, 245, 61),
}


def interpolated_shape(shapes, frame_number):
    frame_numbers = [shape["frame"] for shape in shapes]
    index = bisect_right(frame_numbers, frame_number) - 1
    if index < 0:
        return None
    if index >= len(shapes):
        return None

    current = shapes[index]
    if current["outside"]:
        return None
    if index == len(shapes) - 1:
        return current["type"], current["points"]

    following = shapes[index + 1]
    if (
        current["type"] != following["type"]
        or len(current["points"]) != len(following["points"])
    ):
        return current["type"], current["points"]

    fraction = (frame_number - current["frame"]) / (
        following["frame"] - current["frame"]
    )
    points = [
        first + (second - first) * fraction
        for first, second in zip(current["points"], following["points"])
    ]
    return current["type"], points


def draw_track(frame, track, frame_number):
    shape = interpolated_shape(track["shapes"], frame_number)
    if shape is None:
        return

    shape_type, points = shape
    color = COLORS.get(track["label"], (0, 255, 255))
    if shape_type == "rectangle":
        x1, y1, x2, y2 = map(round, points)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
        cv2.putText(
            frame,
            track["label"],
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
    elif shape_type == "polyline":
        line_points = [
            (round(points[index]), round(points[index + 1]))
            for index in range(0, len(points), 2)
        ]
        for start, end in zip(line_points, line_points[1:]):
            cv2.line(frame, start, end, color, 3, cv2.LINE_AA)
        x, y = line_points[0]
        cv2.putText(
            frame,
            track["label"],
            (x, max(24, y - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
    elif shape_type == "points":
        x, y = map(round, points[:2])
        cv2.drawMarker(
            frame,
            (x, y),
            color,
            cv2.MARKER_CROSS,
            18,
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            track["label"],
            (x + 12, max(24, y - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        default="run4",
        help="Run identifier used in input/output filenames (default: run4).",
    )
    parser.add_argument(
        "--video", type=Path, help="Override input/<run>/<run>_video.mp4."
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        help="Override input/<run>/<run>_annotations.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Override output/<run>_annotated_video.mp4.",
    )
    parser.add_argument("--max-frames", type=int)
    return parser.parse_args()


def resolve_paths(arguments):
    input_directory = Path("input") / arguments.run
    video = arguments.video or input_directory / f"{arguments.run}_video.mp4"
    annotations = (
        arguments.annotations or input_directory / f"{arguments.run}_annotations.json"
    )
    output = arguments.output or Path("output") / f"{arguments.run}_annotated_video.mp4"
    return video, annotations, output


def main():
    arguments = parse_arguments()
    video_path, annotations_path, output_path = resolve_paths(arguments)
    tracks = json.loads(annotations_path.read_text(encoding="utf-8"))[0]["tracks"]

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    frame_rate = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter.fourcc(*"mp4v"),
        frame_rate,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create output: {output_path}")

    frame_number = 0
    while arguments.max_frames is None or frame_number < arguments.max_frames:
        success, frame = capture.read()
        if not success:
            break
        for track in tracks:
            draw_track(frame, track, frame_number)
        writer.write(frame)
        frame_number += 1

    capture.release()
    writer.release()
    print(f"Wrote {frame_number} frames to {output_path}")


if __name__ == "__main__":
    main()
