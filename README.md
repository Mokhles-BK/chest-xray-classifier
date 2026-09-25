# Chest X-Ray14 Multi-Label Classifier

A multi-label deep learning classifier for the NIH ChestX-ray14 dataset — 15
pathology labels per image, trained end-to-end, served as a REST API, and
containerized with Docker.

Full results, per-class metrics, and the architecture comparison are in
[RESULTS.md](RESULTS.md).

## What this is

- **Data pipeline**: downloads and extracts the ChestX-ray14 dataset
  (resumable — skips work already done on re-run)
- **Two trained baselines**: ResNet18 and EfficientNet-B0, both compared
  honestly with the same training setup (see RESULTS.md)
- **Multi-label metrics done right**: per-class precision/recall/F1 with
  tuned thresholds — never plain accuracy, which is meaningless on an
  imbalanced multi-label task
- **A documented ablation**: tested class-weighted loss (`pos_weight`) to
  address class imbalance, with an honest negative-result writeup
- **A served model**: FastAPI inference endpoint, containerized with Docker

## Project structure

```
scripts/              Data download/extraction, training, re-evaluation
src/chestxray/         Core package: dataset, model, loss, metrics, config, labels
api/                    FastAPI inference service (app.py, schemas.py)
runs/                   Training run outputs (gitignored — configs, checkpoints, metrics)
data/                   Dataset (gitignored — ~7GB after extraction)
Dockerfile              Container build for the inference API
docker-compose.yml      Runs the API with a model checkpoint mounted at runtime
RESULTS.md              Full results, per-class metrics, architecture comparison
```

## Setup

```bash
pip install torch torchvision timm pandas pillow pyarrow
```

For GPU training, install the CUDA build matching your driver from
[pytorch.org](https://pytorch.org/get-started/locally/) instead of the line
above.

## Getting the data

```bash
python scripts/download_data.py
python scripts/extract_images.py
```

Set `CHESTXRAY_DATA_ROOT` to point these at a different location (e.g. a
mounted Google Drive path) if you don't want the data living inside the repo
folder.

## Training

```bash
python scripts/train_baseline.py --backbone resnet18 --epochs 20 --batch-size 16
```

Swap `--backbone` for `efficientnet_b0` (or any `timm`-supported model name)
to train a different architecture. Use `--smoke-test` first to verify the
pipeline runs end-to-end on a small subset before committing to a full run.

Each run writes to `runs/<run_id>/`: `config.json`, `metrics.csv`, `run.json`
(includes the tuned per-class thresholds), and `best_model.pt`.

## Re-evaluating an existing checkpoint

No need to retrain to see threshold-tuned vs fixed-threshold metrics for a
run you already have:

```bash
python scripts/reevaluate.py --run-dir runs/<run_id> --backbone <backbone_name>
```

## Running the API

Locally:
```bash
pip install fastapi uvicorn python-multipart
MODEL_DIR=runs/<run_id> uvicorn api.app:app --reload
```
Then open `http://127.0.0.1:8000/docs` for an interactive test page.

With Docker:
```bash
docker compose up --build
```
Edit `docker-compose.yml` to point at whichever run directory you want to
serve. Model weights are mounted at runtime, never baked into the image or
committed to git.

## Results

See [RESULTS.md](RESULTS.md) for the full architecture comparison, per-class
metrics for both models, and the `pos_weight` ablation writeup.
