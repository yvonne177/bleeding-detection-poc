from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from src.flow.base import DenseFlowEstimator


class NeuFlowV2Estimator(DenseFlowEstimator):
    """Official NeuFlow v2 adapter returning dx/dy at the original ROI resolution."""

    def __init__(self, model_config: dict, device_name: str) -> None:
        requested_device = torch.device(device_name)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available. Use --device cpu or install a CUDA-enabled PyTorch build.")
        self.device = requested_device
        self.input_width = int(model_config["input"]["width"])
        self.input_height = int(model_config["input"]["height"])
        if self.input_width % 16 or self.input_height % 16:
            raise ValueError("NeuFlow input width and height must both be divisible by 16.")

        repository_path = model_config.get("model", {}).get("repository_path")
        if repository_path:
            repository = Path(repository_path).expanduser().resolve()
            if not repository.is_dir():
                raise FileNotFoundError(f"NeuFlow repository_path does not exist: {repository}")
            sys.path.insert(0, str(repository))

        try:
            from NeuFlow.neuflow import NeuFlow
        except ImportError as error:
            raise ImportError(
                "NeuFlow v2 is not importable. Clone the official repository and either run this "
                "from its environment or set model.repository_path in configs/model_neuflow_v2.yaml."
            ) from error

        checkpoint_path = Path(model_config["checkpoint"]["path"]).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"NeuFlow checkpoint not found: {checkpoint_path}")

        self.model = NeuFlow().to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.eval()
        self.use_half = self.device.type == "cuda"
        if self.use_half:
            self.model.half()
        self.model.init_bhwd(1, self.input_height, self.input_width, str(self.device), amp=self.use_half)

    def estimate(self, first_roi: np.ndarray, second_roi: np.ndarray) -> np.ndarray:
        if first_roi.shape != second_roi.shape:
            raise ValueError("Consecutive ROI crops must have matching dimensions.")
        roi_height, roi_width = first_roi.shape[:2]
        first = self._to_tensor(first_roi)
        second = self._to_tensor(second_roi)
        with torch.inference_mode():
            flow = self.model(first, second)[-1][0].float().permute(1, 2, 0).cpu().numpy()
        flow = cv2.resize(flow, (roi_width, roi_height), interpolation=cv2.INTER_LINEAR)
        flow[..., 0] *= roi_width / self.input_width
        flow[..., 1] *= roi_height / self.input_height
        return flow.astype(np.float32, copy=False)

    def _to_tensor(self, roi: np.ndarray) -> torch.Tensor:
        resized = cv2.resize(roi, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(np.ascontiguousarray(resized)).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return tensor.half() if self.use_half else tensor.float()
