# Clash Farmer

A Windows desktop prototype for detecting resource collectors in Clash of Clans screenshots and inspecting possible Sneaky Goblin deployment routes.

The project combines computer vision, OCR, a Tkinter dashboard, and a state-based controller for Google Play Games on PC. Its main working workflow is **saved-image analysis**: detect collectors, estimate obstacles and deployment boundaries, and explain why targets are selected or skipped.

> **Status:** Experimental. Offline analysis and controller simulation work, but a complete live attack and unattended farming have not been verified. Live attacks remain blocked by the geometry validation check. Jump Spell deployment is deferred.

## Features

- YOLO-based detection of gold mines, elixir collectors, and dark elixir drills.
- Experimental wall, building, and deployment-boundary models.
- Grid-based route proposals that exclude obstructed, hidden, and clipped targets.
- OCR for available loot, troop counts, and battle results.
- A dashboard with numbered deployment markers, current activity, evidence, and action history.
- Dataset annotation, import, augmentation, training, and evaluation tools.
- A controller with state checks, troop budgets, camera checks, cancellation, and deployment-count verification.

## How it works

1. Load a saved screenshot or explicitly start window monitoring.
2. Detect collectors and estimate the deployment boundary and obstacles.
3. Calculate candidate routes and apply the exposed-collector attack policy.
4. Display deployment proposals with confidence scores and skip reasons.
5. For a calibrated live run, verify the screen, camera position, target, and troop count before input. Verify troop consumption afterward.

Saved-image proposals are marked `proposal_only` and cannot be executed by the controller. A missing collector alone does not establish destruction: the runtime check also requires matching rubble at its previous location.

## Setup

Windows is required for live window capture and input. Development was tested with Python 3.13. Install Tesseract OCR with English language data separately.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-ml.txt
Copy-Item config.example.json config.json
python dashboard.py
```

Alternatively, use `launch.bat`. It creates `config.json` from the example if needed. Set the exact game window title, Tesseract executable path, and capture dimensions in your local configuration. The included calibration targets a 1920 x 1080 window and may need adjustment for another setup.

**Model weights and datasets are not included in this source repository.** Train or supply compatible checkpoints before using detector-backed analysis. The example configuration lists the expected local paths. CPU inference is the portable default; CUDA requires a compatible PyTorch installation and device setting.

Opening the dashboard starts no game capture or input. **Explain saved base** analyzes an existing PNG or JPG. **Start monitoring** explicitly starts window capture and sends no mouse input.

```powershell
python dashboard.py --explain path\to\scouting-screenshot.png
python farmer.py --list-windows
```

The Progress tab shows the overlay and current action. The Activity tab shows observations, measured evidence, and upcoming actions. Settings control loot thresholds, troop budgets, batch sizes, and timeouts. Use Stop to cancel an active worker.

## Training and evaluation

Use the annotation editor to review collector boxes and group related screenshots by base. Keep frames of the same base, including video frames, in a single split. Partially labeled images must not be treated as fully reviewed training examples.

```powershell
python train_detector.py --data data/collectors --classes collectors --prepare-only
python train_detector.py --data data/collectors --classes collectors --model yolo11n.pt --device cpu
python prepare_recording.py path\to\recording.mp4 --output data/video-review/session
```

`train_detector.py` also accepts a prepared dataset with `--dataset-yaml`. The wall and boundary training scripts use local reference annotations. Recording replays require inspected labels for the corresponding recording; example defaults in development utilities refer to local assets excluded from Git.

The original collector model was evaluated on four held-out bases containing 68 boxes: mAP50 was 0.9666 and mAP50-95 was 0.8599. This is a small detector evaluation, not an attack-success rate. A video fine-tune reduced validation recall, so the stronger original detector remained active.

## Development

```powershell
python -m pip install -r requirements-dev.txt
python -m unittest discover
ruff check .
ruff format --check .
```

Tests cover dataset splitting, OCR agreement, state recognition, routing, troop budgets, cancellation, and a simulated attack flow. They do not prove live gameplay reliability.

| Module | Responsibility |
| --- | --- |
| `dashboard.py`, `dashboard_worker.py` | Desktop controls, worker lifecycle, and activity display |
| `farmer.py`, `session_runner.py` | State-based controller and session limits |
| `vision.py`, `deployment_map.py` | Detections, geometry, and route planning |
| `neural_detector.py`, `wall_model.py`, `boundary_model.py` | Model inference |
| `army.py`, `ui_numbers.py`, `state_recognition.py` | Troop counts, OCR, and UI recognition |
| `window_capture.py` | Window capture and coordinate mapping |
| `dataset.py`, `train_detector.py` | Reviewed annotations, exports, and model training |

## Limitations

Game pathfinding, scenery, zoom, animation, and building levels can invalidate a visual route estimate. Collector false positives and incomplete wall masks remain possible. Unknown screens stop the controller; automatic reconnection and login recovery are not implemented. The project is not ready for 24/7 operation.

## Data and attribution

Some training experiments used [Clash of clans villages, version 28](https://universe.roboflow.com/cocsegmentation/clash-of-clans-villages/dataset/28), published by Cocsegmentation under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Processing filtered village frames and collector classes, renamed classes, converted COCO boxes to YOLO labels, removed exact duplicates, and corrected inspected labels. Imported exports retain their original notices locally.

Clash of Clans artwork and game assets belong to their respective owners. This is an independent project with no affiliation to Supercell. Third-party packages retain their own licenses.

## Preparing a GitHub upload

`.gitignore` excludes local configuration, datasets, checkpoints, recordings, generated images, logs, and calibration records. Keep those exclusions when adding your remote repository.

```powershell
python prepare_github.py
```

This creates a source-only upload folder and ZIP under `dist/`. It includes this README, Python source and tests, dependency/tooling files, a generic configuration example, and the small runtime template images. It does not publish anything or delete local training assets.
