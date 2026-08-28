from __future__ import annotations

import argparse
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
from src.scoring.gravity_alignment import gravity_alignment
from src.scoring.magnitude_score import candidate_mask, flow_magnitude
from src.visualization.render_video import (
    draw_cvat_context,
    flow_to_hsv_bgr,
    full_frame_overlay,
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
    parser.add_argument("--start-frame", type=int, help="Override the first CVAT-annotated frame.")
    parser.add_argument("--end-frame", type=int, help="Override the exclusive CVAT-annotated ending frame.")
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

    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open input video: {args.input}")
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    annotations = None
    if args.cvat_annotations is not None:
        annotations = CvatAnnotations(args.cvat_annotations, args.cvat_task)
        annotated_start, annotated_end = annotations.annotated_frame_range()
        start_frame = args.start_frame if args.start_frame is not None else annotated_start
        end_frame = args.end_frame if args.end_frame is not None else annotated_end
    else:
        start_frame = args.start_frame if args.start_frame is not None else 0
        end_frame = args.end_frame if args.end_frame is not None else frame_count
    if start_frame < 0 or end_frame <= start_frame or end_frame > frame_count:
        raise ValueError(f"Select frames within [0, {frame_count}] with an end frame greater than the start frame.")
    if annotations is not None:
        bbox = annotations.fixed_roi_for_range(
            start_frame, end_frame, frame_width, frame_height, args.cvat_roi_padding
        )
        print(f"CVAT annotated frame range: [{annotated_start}, {annotated_end})")
        print(f"Selected source-frame range: [{start_frame}, {end_frame})")
        print(f"Fixed NeuFlow ROI from CVAT bleeding-area track: {bbox}")
    else:
        bbox = BoundingBox.from_config(trial_config["roi"]["bbox"])
    estimator = NeuFlowV2Estimator(model_config, args.device)
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ok, previous_frame = capture.read()
    if not ok:
        raise ValueError("No readable frame at --start-frame.")
    previous_roi = bbox.crop(previous_frame)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    flow_directory = args.output.parent / f"{args.output.stem}_flows"
    if args.save_flow:
        flow_directory.mkdir(parents=True, exist_ok=True)
    dashboard_writer = None
    overlay_writer = None
    overlay_path = args.output.with_name(f"{args.output.stem}_overlay.mp4")
    latencies_ms: list[float] = []
    processed_frames = 0
    source_index = start_frame + 1

    while source_index < end_frame:
        ok, current_frame = capture.read()
        if not ok:
            break
        current_roi = bbox.crop(current_frame)
        if estimator.device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        flow = estimator.estimate(previous_roi, current_roi)
        if estimator.device.type == "cuda":
            torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - started) * 1000.0)

        magnitude = flow_magnitude(flow)
        mask = candidate_mask(magnitude, float(scoring["magnitude_threshold"]))
        gravity = scoring.get("gravity", {})
        if gravity.get("enabled", False):
            vector = gravity["vector"]
            alignment = gravity_alignment(flow, float(vector["x"]), float(vector["y"]))
            mask &= alignment >= float(gravity["minimum_alignment"])
        active_bleeding_box = None
        origin_shapes = []
        if annotations is not None:
            active_bleeding_box = annotations.bleeding_box_at(source_index)
            annotation_mask = annotations.bleeding_mask_for_roi(source_index, bbox)
            mask &= annotation_mask if annotation_mask is not None else False
            origin_shapes = annotations.origin_shapes_at(source_index)

        flow_bgr = flow_to_hsv_bgr(flow, magnitude_clip)
        magnitude_bgr = magnitude_to_bgr(magnitude, magnitude_clip)
        annotated_frame = draw_cvat_context(current_frame, active_bleeding_box, origin_shapes)
        dashboard = render_dashboard(annotated_frame, current_roi, flow_bgr, magnitude_bgr, mask, bbox)
        if dashboard_writer is None:
            height, width = dashboard.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*visualization.get("output_codec", "mp4v"))
            dashboard_writer = cv2.VideoWriter(str(args.output), fourcc, source_fps, (width, height))
            if not dashboard_writer.isOpened():
                raise RuntimeError(f"Cannot create output video: {args.output}")
            if args.save_overlay:
                overlay_writer = cv2.VideoWriter(str(overlay_path), fourcc, source_fps, (current_frame.shape[1], current_frame.shape[0]))
        dashboard_writer.write(dashboard)
        if overlay_writer is not None:
            overlay = full_frame_overlay(annotated_frame, flow_bgr, mask, bbox, float(visualization["overlay_alpha"]))
            overlay_writer.write(overlay)
        if args.save_flow:
            np.savez_compressed(flow_directory / f"flow_{source_index:06d}.npz", flow=flow)

        previous_roi = current_roi
        processed_frames += 1
        source_index += 1

    capture.release()
    if dashboard_writer is not None:
        dashboard_writer.release()
    if overlay_writer is not None:
        overlay_writer.release()
    if not latencies_ms:
        raise ValueError("No consecutive frames were processed. Increase the selected frame range.")
    print(f"Source video FPS: {source_fps:.2f}")
    print(f"Processed frame pairs: {processed_frames}")
    print(latency_summary(latencies_ms))
    print(f"Dashboard video: {args.output}")
    if args.save_overlay:
        print(f"Full-frame overlay: {overlay_path}")


if __name__ == "__main__":
    main()
