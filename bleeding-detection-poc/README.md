# NeuFlow v2 ROI Optical-Flow Pipeline

This project runs pretrained NeuFlow v2 on video inside a fixed region of interest (ROI). It measures **qualitative optical-flow behavior and runtime**; it is not a general-purpose detector or an accuracy benchmark. Adapt the ROI, scoring rules, annotations, and visualization settings to your video domain.

## What it does

For every consecutive frame pair, the fixed bounding box is cropped from the original frame **before** NeuFlow inference. The two crops are resized to NeuFlow's configured input size, inferred, and the dense flow is resized back to the original crop size with separate horizontal and vertical displacement scaling. Flow uses `H x W x 2` arrays where `flow[..., 0]` is `dx` rightward and `flow[..., 1]` is `dy` downward.

The dashboard MP4 shows the full source frame with its ROI, then the cropped ROI, HSV optical flow, magnitude map, and candidate mask. Hue encodes direction; saturation/value increase with flow magnitude. Candidate mask pixels satisfy the configured magnitude threshold, optionally with the configured directional-alignment threshold. This is candidate motion only, not a confirmed event or class prediction.

When compatible CVAT inputs are supplied, the yellow rectangle is the one fixed NeuFlow crop for the selected interval. Annotated regions can gate the candidate mask, while context shapes can be displayed on the output. These annotations guide visualization and ROI selection only; they are not used to calculate accuracy.

## Install

Create the Python environment and install the base packages:

```powershell
$PROJECT_DIR = "C:\path\to\project"
Set-Location $PROJECT_DIR
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the commands below from the project root: the directory that contains `scripts`, `configs`, `data`, and `src`. If the checkout has a wrapper directory around that project root, change into the inner directory first and reference the environment with its relative path.

For every new PowerShell terminal, run project commands through `.venv\Scripts\python.exe`. This does not require activating the environment, so it works when PowerShell blocks `Activate.ps1` scripts:

```powershell
$PROJECT_DIR = "C:\path\to\project"
Set-Location $PROJECT_DIR
.\.venv\Scripts\python.exe scripts\run_neuflow.py --help
```

To activate the environment for the current terminal session instead, first allow scripts only for that process:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

NeuFlow v2 is used from its upstream repository. Clone it alongside this project, or set `model.repository_path` in the model configuration to an existing checkout:

```powershell
cd ..
git clone https://github.com/neufieldrobotics/NeuFlow_v2.git
Set-Location $PROJECT_DIR
```

Use the PyTorch and checkpoint versions supported by the upstream NeuFlow repository. Set `model.repository_path` in the model configuration when the checkout is elsewhere.

## Checkpoint

Place a compatible NeuFlow checkpoint at the path configured by `checkpoint.path`, or update that setting in the model configuration. The checkpoint location is not hardcoded in Python.

## Configure

Edit the trial configuration to set the fixed `roi.bbox` in original-frame pixels. It must be fully inside the source video frame and remains unchanged for every frame pair. Configure the NeuFlow resize resolution in the model configuration; both dimensions must be divisible by 16.

Set the candidate magnitude threshold, optional downward gravity vector, minimum alignment, visualization magnitude clip, and overlay opacity in the trial config.

## Run

```powershell
.\.venv\Scripts\python.exe scripts\run_neuflow.py `
  --input path/to/input.mp4 `
  --output path/to/output.mp4 `
  --config path/to/trial_config.yaml `
  --model-config path/to/model_config.yaml `
  --device cuda `
  --start-frame 0 `
  --end-frame 300 `
  --save-flow `
  --save-overlay
```

Use `--device cpu` when CUDA is unavailable. Frame arguments are zero-based source-video indices: `--start-frame N` selects the first source frame, and `--end-frame M` is exclusive. The runner reads frame `N` as the previous frame and writes flow for source frames `N + 1` through `M - 1`, producing `M - N - 1` consecutive pairs when all reads succeed. For example, `--start-frame 100 --end-frame 250` processes pairs `(100,101)` through `(248,249)` and writes 149 output frames. Without CVAT, the default range is the complete video. With CVAT, the default range is the annotated range. `--save-flow` writes ROI-native dense flow arrays as compressed NPZ files, and `--save-overlay` creates a second full-frame MP4 with colored candidate flow only inside the bounding box.

## Run With CVAT Files

For a project using the runner's standard naming convention, pass an identifier. The runner resolves the video, optional CVAT export, optional CVAT task file, and result path automatically. Otherwise, pass explicit `--input`, `--output`, and annotation paths.

```powershell
.\.venv\Scripts\python.exe scripts\run_neuflow.py `
  --run example-id `
  --device cuda `
  --cvat-roi-padding 16 `
  --save-flow `
  --save-overlay
```

With CVAT annotations, the automatic range is the annotated interval. Add `--start-frame` and `--end-frame` to override it for a shorter trial. Keep exploratory intervals long enough to contain useful motion while respecting the available compute. The fixed ROI is computed once from annotations in the selected interval, so flow is inferred on the ROI rather than on a full frame and cropped afterward.

## Runtime output

The console reports source video FPS, processed frame pairs, and NeuFlow-only mean, median, p95 latency, and inference FPS. A frame manifest records the requested range and the exact source frame index for every output frame. Timing starts immediately before the model adapter runs and stops after CUDA synchronization (when applicable), so video decoding, cropping, scoring, disk I/O, and rendering are excluded.

## Limitations

Optical flow reflects apparent motion, including camera or object movement, scene deformation, lighting changes, reflections, and fluid motion. Without representative ground truth, this pipeline cannot establish accuracy or operational reliability. Review visual outputs qualitatively and use runtime figures only as hardware- and configuration-specific benchmarks.
