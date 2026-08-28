from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.flow.neuflow_v2 import NeuFlowV2Estimator
from src.preprocessing.bbox_crop import BoundingBox
from src.preprocessing.cvat_annotations import CvatAnnotations
from src.scoring.blood_appearance import blood_color_mask, darkening_mask
from src.scoring.gravity_alignment import gravity_alignment
from src.scoring.magnitude_score import candidate_mask, flow_magnitude
from src.visualization.render_video import (
    draw_cvat_context,
    flow_to_hsv_bgr,
    full_frame_overlay,
    mask_to_bgr,
    magnitude_to_bgr,
    render_dashboard,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROI-first NeuFlow v2 optical-flow POC for surgical video.")
    parser.add_argument("--run", help="Run identifier, such as run4, used to resolve standard input and output paths.")
    parser.add_argument("--input", type=Path, help="Override source surgical video path.")
    parser.add_argument("--output", type=Path, help="Override dashboard output MP4 path.")
    parser.add_argument("--config", type=Path, default=Path("configs/trial_config.yaml"), help="Trial YAML configuration.")
    parser.add_argument("--model-config", type=Path, default=Path("configs/model_neuflow_v2.yaml"), help="NeuFlow YAML configuration.")
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device; defaults to CUDA when available, otherwise CPU.",
    )
    parser.add_argument("--start-frame", type=int, help="First source frame to export; defaults to 0. Annotations stay keyed to source frame indices.")
    parser.add_argument("--end-frame", type=int, help="Exclusive last source frame to export; defaults to the full video length.")
    parser.add_argument("--save-flow", action="store_true", help="Save each ROI flow as compressed NPZ.")
    parser.add_argument("--save-overlay", action="store_true", help="Also save an original-frame overlay MP4.")
    parser.add_argument("--cvat-annotations", type=Path, help="CVAT JSON export with bleeding-area and blood-origin tracks.")
    parser.add_argument("--cvat-task", type=Path, help="Optional matching CVAT task JSON used to validate labels.")
    parser.add_argument("--cvat-roi-padding", type=int, default=0, help="Pixels of padding around the fixed ROI derived from CVAT bleeding areas.")
    return parser.parse_args()


def resolve_run_paths(args: argparse.Namespace) -> None:
    if args.run:
        args.input = args.input or PROJECT_ROOT / "data" / "raw_videos" / f"{args.run}_video.mp4"
        args.output = args.output or PROJECT_ROOT / "results" / f"{args.run}_neuflow.mp4"
        args.cvat_annotations = args.cvat_annotations or PROJECT_ROOT / "data" / "annotations" / "cvat_exports" / f"{args.run}_annotations.json"
        args.cvat_task = args.cvat_task or PROJECT_ROOT / "data" / "annotations" / "cvat_tasks" / f"{args.run}_task.json"
    if args.input is None or args.output is None:
        raise ValueError("Provide --run, or provide both --input and --output.")


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def resolve_model_paths(model_config: dict) -> dict:
    checkpoint = Path(model_config["checkpoint"]["path"])
    if not checkpoint.is_absolute():
        model_config["checkpoint"]["path"] = str(PROJECT_ROOT / checkpoint)
    repository = model_config.get("model", {}).get("repository_path")
    if repository and not Path(repository).is_absolute():
        model_config["model"]["repository_path"] = str(PROJECT_ROOT / repository)
    return model_config


def latency_summary(latencies_ms: list[float]) -> str:
    values = np.asarray(latencies_ms, dtype=np.float64)
    fps = 1000.0 / values.mean()
    return (
        f"NeuFlow inference: mean={values.mean():.2f} ms, median={np.median(values):.2f} ms, "
        f"p95={np.percentile(values, 95):.2f} ms, FPS={fps:.2f}"
    )


def write_frame_manifest(
    *,
    args: argparse.Namespace,
    overlay_path: Path,
    source_fps: float,
    start_frame: int,
    end_frame: int,
    written_source_frames: list[int],
) -> Path:
    manifest_path = args.output.with_name(f"{args.output.stem}_frame_manifest.json")
    manifest = {
        "description": (
            "Output frame i of the dashboard and overlay videos corresponds to "
            "source_frames[i] in the source video, which equals CVAT annotation frame "
            "source_frames[i]. With the default full-video range, source_frames[i] == i."
        ),
        "source_video": str(args.input),
        "dashboard_video": str(args.output),
        "overlay_video": str(overlay_path) if args.save_overlay else None,
        "source_fps": source_fps,
        "requested_start_frame": start_frame,
        "requested_end_frame": end_frame,
        "output_frame_count": len(written_source_frames),
        "source_frames": written_source_frames,
    }
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    return manifest_path


def main() -> None:
    args = parse_args()
    resolve_run_paths(args)
    if args.cvat_roi_padding < 0:
        raise ValueError("Use a non-negative --cvat-roi-padding.")
    if args.cvat_task is not None and args.cvat_annotations is None:
        raise ValueError("--cvat-task requires --cvat-annotations.")
    trial_config = load_yaml(args.config)
    model_config = resolve_model_paths(load_yaml(args.model_config))
    if not trial_config["roi"].get("enabled", False):
        raise ValueError("This ROI-first POC requires roi.enabled: true.")

    scoring = trial_config["scoring"]
    visualization = trial_config["visualization"]
    magnitude_clip = float(visualization["flow_magnitude_clip"])
    blood_config = scoring.get("blood_appearance", {})
    color_config = blood_config.get("color", {})
    darkening_config = blood_config.get("darkening", {})

    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open input video: {args.input}")
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    annotations = None
    annotated_start = annotated_end = None
    if args.cvat_annotations is not None:
        annotations = CvatAnnotations(args.cvat_annotations, args.cvat_task)
        annotated_start, annotated_end = annotations.annotated_frame_range()
    # Default to the whole source video so output frame i stays equal to source frame i.
    start_frame = args.start_frame if args.start_frame is not None else 0
    end_frame = args.end_frame if args.end_frame is not None else frame_count
    if start_frame < 0 or end_frame <= start_frame or end_frame > frame_count:
        raise ValueError(f"Select frames within [0, {frame_count}] with an end frame greater than the start frame.")
    if annotations is not None:
        bbox = annotations.fixed_roi_for_range(
            annotated_start, annotated_end, frame_width, frame_height, args.cvat_roi_padding
        )
        print(annotations.describe_frame_mapping())
        print(f"CVAT annotated frame range in source frames: [{annotated_start}, {annotated_end})")
        print(f"Exported source-frame range: [{start_frame}, {end_frame})")
        print(f"Fixed NeuFlow ROI from CVAT bleeding-area track: {bbox}")
        if annotated_end > frame_count:
            print(f"Warning: annotations reference source frame {annotated_end - 1}, past the last video frame {frame_count - 1}.")
    else:
        bbox = BoundingBox.from_config(trial_config["roi"]["bbox"])
    estimator = NeuFlowV2Estimator(model_config, args.device)
    # Read sequentially from frame 0 instead of seeking: compressed-video seeks can
    # snap to the nearest keyframe, shifting frame content relative to CVAT frame indices.
    for _ in range(start_frame):
        ok, _ = capture.read()
        if not ok:
            raise ValueError("Video ended before reaching --start-frame.")
    previous_roi = None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    flow_directory = args.output.parent / f"{args.output.stem}_flows"
    if args.save_flow:
        flow_directory.mkdir(parents=True, exist_ok=True)
    dashboard_writer = None
    overlay_writer = None
    overlay_path = args.output.with_name(f"{args.output.stem}_overlay.mp4")
    latencies_ms: list[float] = []
    processed_frames = 0
    written_source_frames: list[int] = []
    source_index = start_frame

    while source_index < end_frame:
        ok, current_frame = capture.read()
        if not ok:
            break
        current_roi = bbox.crop(current_frame)
        if previous_roi is None:
            # First emitted frame has no predecessor, so report zero flow instead of dropping it.
            flow = np.zeros((current_roi.shape[0], current_roi.shape[1], 2), dtype=np.float32)
        else:
            if estimator.device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            flow = estimator.estimate(previous_roi, current_roi)
            if estimator.device.type == "cuda":
                torch.cuda.synchronize()
            latencies_ms.append((time.perf_counter() - started) * 1000.0)
            processed_frames += 1

        magnitude = flow_magnitude(flow)
        mask = candidate_mask(magnitude, float(scoring["magnitude_threshold"]))
        gravity = scoring.get("gravity", {})
        if gravity.get("enabled", False):
            vector = gravity["vector"]
            alignment = gravity_alignment(flow, float(vector["x"]), float(vector["y"]))
            mask &= alignment >= float(gravity["minimum_alignment"])

        # Appearance gates keep dark-red bleeding and reject instruments, which move fast but are never dark red.
        color_mask = np.zeros(magnitude.shape, dtype=bool)
        if color_config.get("enabled", False):
            color_mask = blood_color_mask(
                current_roi,
                float(color_config.get("hue_margin", 12.0)),
                float(color_config.get("minimum_saturation", 90.0)),
                float(color_config.get("maximum_value", 150.0)),
                int(color_config.get("opening_kernel_size", 3)),
            )
            mask &= color_mask
        darkened_mask = np.zeros(magnitude.shape, dtype=bool)
        if darkening_config.get("enabled", False):
            darkened_mask = darkening_mask(
                previous_roi,
                current_roi,
                float(darkening_config.get("minimum_value_drop", 2.0)),
                int(darkening_config.get("blur_kernel_size", 5)),
            )
            mask &= darkened_mask
        active_bleeding_box = None
        origin_shapes = []
        if annotations is not None:
            active_bleeding_box = annotations.bleeding_box_at(source_index)
            annotation_mask = annotations.bleeding_mask_for_roi(source_index, bbox)
            mask &= annotation_mask if annotation_mask is not None else False
            origin_shapes = annotations.origin_shapes_at(source_index)

        flow_bgr = flow_to_hsv_bgr(flow, magnitude_clip)
        magnitude_bgr = magnitude_to_bgr(magnitude, magnitude_clip)
        color_mask_bgr = mask_to_bgr(color_mask)
        darkened_mask_bgr = mask_to_bgr(darkened_mask)
        annotated_frame = draw_cvat_context(current_frame, active_bleeding_box, origin_shapes)
        dashboard = render_dashboard(
            annotated_frame,
            current_roi,
            flow_bgr,
            magnitude_bgr,
            color_mask_bgr,
            darkened_mask_bgr,
            mask,
            bbox,
        )
        if dashboard_writer is None:
            height, width = dashboard.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*visualization.get("output_codec", "mp4v"))
            dashboard_writer = cv2.VideoWriter(str(args.output), fourcc, source_fps, (width, height))
            if not dashboard_writer.isOpened():
                raise RuntimeError(f"Cannot create output video: {args.output}")
            if args.save_overlay:
                overlay_writer = cv2.VideoWriter(str(overlay_path), fourcc, source_fps, (current_frame.shape[1], current_frame.shape[0]))
        dashboard_writer.write(dashboard)
        written_source_frames.append(source_index)
        if overlay_writer is not None:
            overlay = full_frame_overlay(annotated_frame, flow_bgr, mask, bbox, float(visualization["overlay_alpha"]))
            overlay_writer.write(overlay)
        if args.save_flow:
            np.savez_compressed(flow_directory / f"flow_{source_index:06d}.npz", flow=flow)

        previous_roi = current_roi
        source_index += 1

    capture.release()
    if dashboard_writer is not None:
        dashboard_writer.release()
    if overlay_writer is not None:
        overlay_writer.release()
    if not latencies_ms:
        raise ValueError("No consecutive frames were processed. Increase the selected frame range.")
    manifest_path = write_frame_manifest(
        args=args,
        overlay_path=overlay_path,
        source_fps=source_fps,
        start_frame=start_frame,
        end_frame=end_frame,
        written_source_frames=written_source_frames,
    )
    print(f"Source video FPS: {source_fps:.2f}")
    print(f"Processed frame pairs: {processed_frames}")
    print(latency_summary(latencies_ms))
    print(f"Dashboard video: {args.output}")
    if args.save_overlay:
        print(f"Full-frame overlay: {overlay_path}")
    print(f"Frame manifest: {manifest_path}")


if __name__ == "__main__":
    main()
