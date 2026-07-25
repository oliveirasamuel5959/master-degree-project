# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Master's research project: **Urban Threat Anticipation using Computer Vision and Temporal Deep Learning**. The research goal is to *anticipate* criminal activity from behavioral patterns in surveillance video (before the event), rather than detect it after the fact. See `README.md`, `docs/mission.md`, and `docs/architecture.md` for the full research framing and the intended end-to-end pipeline (YOLO detection → ByteTrack tracking → feature extraction → LSTM temporal modeling → self-attention → threat prediction).

The code in `src/` is currently at an early phase: video action-classification baselines on public datasets (UCF-Crime, etc.). The full detection/tracking/temporal pipeline described in the docs is **not yet implemented** — treat `docs/` as the roadmap, `src/` as the current reality.

## Environment & Commands

Python 3.12, managed with **uv** (`uv.lock`, `pyproject.toml`). PyTorch + OpenCV + torchvision stack.

```bash
uv sync                      # install all deps (incl. dev + lint groups)
uv run <cmd>                 # run any command inside the project venv
uv run ruff check .          # lint
uv run ruff format .         # format
uv run pytest                # run tests (note: no real tests exist yet)
```

**Running training/experiments** — modules use absolute `from src....` imports, so run them as modules from the repo root, not as file paths:

```bash
uv run python -m src.experiments.exp_001_training
```

`exp_001_training.py` is the current working entry point (it has its own `main()` with hardcoded paths/hyperparameters). `main.py` at the repo root is a placeholder stub.

## Architecture of the current code

The training flow in `src/experiments/exp_001_training.py` wires together:

- **`src/data/loaders.py`** — `read_train_test_data_txt(path, data_type)` reads dataset split files (e.g. `Anomaly_Train.txt`) into lists of `"ClassName/file.mp4"` strings.
- **`src/data/dataset_preprocessing.py`** — `VideoDataset` (a `torch.utils.data.Dataset`) that samples `num_frames` uniformly-spaced frames per video with OpenCV, applies per-frame transforms, and returns tensors shaped `(C, T, H, W)`. Class labels are derived from the first path segment. `video_preprocessing()` builds the torchvision transform (resize + normalize).
- **`src/models/Models.py`** — `Video3DCNN`, a 3D-CNN built from stacked `Conv3DBlock`s + adaptive pooling + classifier head. This is the active model. `src/models/sample_models.py` holds experimental/sample architectures.
- **`src/utils/training_utils.py`** — `training_validation()` (eval loop) and `save_history_and_plots()` (writes `*_.csv` + accuracy/loss PNGs to the output dir).
- **`src/core/logger.py`** — `get_logger()` returns a Rich-backed logger that also writes daily plain-text logs to `logs/YYYYMMDD.log`; `log_task()` is a spinner context manager. Use this rather than `print` for new code.

There is a **second, separate training abstraction** in `src/train.py` + `src/training/{train_step,test_step}.py` — a reusable `train()` loop with TensorBoard `SummaryWriter` and LR-scheduler support. This is *not* used by `exp_001_training.py` (which has its own inline epoch loop). When adding training code, decide deliberately which of the two paths to build on rather than mixing them.

**Empty/stub files** (scaffolding, not yet implemented): `src/evaluate.py`, `src/tune.py`, `src/serving/inference.py`, `src/experiments/tests.py`, and the `configs/` and `docker/` directories. Don't assume they contain logic.

## Data layout

Datasets live under `data/` (gitignored — not in the repo; must be provided locally). Each dataset follows a `raw/` + `processed/` convention, e.g.:

```
data/UCF_Crime/raw/<ClassName>/<video>.mp4
data/UCF_Crime/processed/Anomaly_Detection_splits/{Anomaly_Train,Anomaly_Test}.txt
```

Other datasets present: `PVSG`, `RLVS`, `TAO_Amodal`, `ViDOR`, `Sample`. Split `.txt` files are the source of truth for what enters a run.

`.gitignore` excludes `data/`, `*.pt`, `*.pth`, and `runs/`. **Model checkpoints (`best_model.pth`) and training outputs are not versioned** — training writes checkpoints to the repo root / `outputs/` and are expected to be reproduced locally.

## Notebooks

`src/notebooks/` is the primary exploration workspace (dataset EDA, dataloader prototyping, brightness analysis, etc.), named `YYYYMMDD_<topic>.ipynb`. Prototyping typically happens in a notebook first, then graduates into `src/` modules.

## Outputs

Experiment artifacts go to `outputs/experiments/<experiment_stem>/` (created automatically), containing `history_.csv`, `acc_plot.png`, `loss_plot.png`. Logs go to `logs/`.
