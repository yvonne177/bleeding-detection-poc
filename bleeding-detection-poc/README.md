# NeuFlow v2 Surgical Bleeding Motion POC

This proof of concept tests pretrained NeuFlow v2 on surgical video. It measures **qualitative optical-flow behavior and runtime**, not bleeding-detection accuracy. There are no hand-annotated masks, so this project intentionally contains no IoU, precision/recall, training, fine-tuning, or alternative optical-flow baselines.

## What it does

For every consecutive frame pair, the fixed bounding box is cropped from the original frame **before** NeuFlow inference. The two crops are resized to NeuFlow's configured input size, inferred, and the dense flow is resized back to the original crop size with separate horizontal and vertical displacement scaling. Flow uses `H x W x 2` arrays where `flow[..., 0]` is `dx` rightward and `flow[..., 1]` is `dy` downward.

The dashboard MP4 shows the full source frame with its ROI, then the cropped ROI, HSV optical flow, magnitude map, and candidate mask. Hue encodes direction; saturation/value increase with flow magnitude. Candidate mask pixels satisfy the configured magnitude threshold, optionally with the configured gravity-alignment threshold. This is candidate motion only, not confirmed bleeding.

## Install

Create the Python environment and install the base packages:

```powershell
cd bleeding-detection-poc
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For every new PowerShell terminal, run project commands through `.venv\Scripts\python.exe`. This does not require activating the environment, so it works when PowerShell blocks `Activate.ps1` scripts:

```powershell
cd C:\Users\yiyua\Downloads\NeuFlow_Test\bleeding-detection-poc
.\.venv\Scripts\python.exe scripts\run_neuflow.py --help
```

To activate the environment for the current terminal session instead, first allow scripts only for that process:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

NeuFlow v2 is used from its official repository. Clone it alongside this project; the supplied model configuration already points to that sibling checkout:

```powershell
cd ..
git clone https://github.com/neufieldrobotics/NeuFlow_v2.git
cd bleeding-detection-poc
```

The official repository documents PyTorch 2.0+ and its pretrained mixed model checkpoint. Set `model.repository_path` in `configs/model_neuflow_v2.yaml` only when your checkout is elsewhere.

## Checkpoint

Place the official `neuflow_mixed.pth` at `checkpoints/neuflow_mixed.pth`, or change only `checkpoint.path` in `configs/model_neuflow_v2.yaml`. The checkpoint location is never hardcoded in Python.

## Configure

Edit `configs/trial_config.yaml` to set the fixed `roi.bbox` in original-frame pixels. It must be fully inside the source video frame and remains unchanged for every frame pair. Configure the NeuFlow resize resolution in `configs/model_neuflow_v2.yaml`; both dimensions must be divisible by 16.

Set the candidate magnitude threshold, optional downward gravity vector, minimum alignment, visualization magnitude clip, and overlay opacity in the trial config.

## Run

```powershell
.\.venv\Scripts\python.exe scripts\run_neuflow.py `
  --input data/raw_videos/example.mp4 `
  --output results/example_neuflow.mp4 `
  --config configs/trial_config.yaml `
  --model-config configs/model_neuflow_v2.yaml `
  --device cuda `
  --start-frame 0 `
  --end-frame 300 `
  --save-flow `
  --save-overlay
```

Use `--device cpu` when CUDA is unavailable. `--end-frame` is exclusive, so the example processes up to 299 consecutive frame pairs. `--save-flow` writes ROI-native dense flow arrays as compressed NPZ files, and `--save-overlay` creates a second full-frame MP4 with colored candidate flow only inside the bounding box.

## Runtime output

The console reports source video FPS, processed frame pairs, and NeuFlow-only mean, median, p95 latency, and inference FPS. Timing starts immediately before the model adapter runs and stops after CUDA synchronization (when applicable), so video decoding, cropping, scoring, disk I/O, and rendering are excluded.

## Limitations

Optical flow reflects apparent motion, including instrument movement, camera motion, smoke, reflections, tissue deformation, and fluid motion. Without ground truth, this POC cannot establish accuracy or clinical reliability. Review the visual outputs qualitatively and use the runtime figures only as hardware- and configuration-specific benchmarks.
