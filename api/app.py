"""FastAPI inference service for the ChestX-ray14 multi-label classifier.

Loads a trained checkpoint + its per-class tuned thresholds (both produced by
scripts/train_baseline.py) and serves predictions over HTTP.

Configure via environment variables:
    MODEL_DIR     Path to a run directory containing best_model.pt + run.json
                  (e.g. runs/resnet18_20260923_023010_49ec6d). Required.

Run locally:
    MODEL_DIR=runs/resnet18_20260923_023010_49ec6d uvicorn api.app:app --reload

Run in Docker: see Dockerfile — MODEL_DIR is expected to be a mounted volume,
since model weights are never committed to the repo.
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from chestxray.dataset import eval_transforms  # noqa: E402
from chestxray.labels import DEFAULT_LABELS  # noqa: E402
from chestxray.model import MultiLabelClassifier  # noqa: E402

from api.schemas import HealthResponse, LabelPrediction, PredictResponse  # noqa: E402

app = FastAPI(title="ChestX-ray14 Multi-Label Classifier", version="1.0")

_state: dict = {}


@app.on_event("startup")
def load_model() -> None:
    model_dir = os.environ.get("MODEL_DIR")
    if not model_dir:
        raise RuntimeError("MODEL_DIR environment variable is required (path to a run directory).")
    model_dir = Path(model_dir)

    run_json_path = model_dir / "run.json"
    ckpt_path = model_dir / "best_model.pt"
    if not run_json_path.exists() or not ckpt_path.exists():
        raise RuntimeError(f"Expected run.json and best_model.pt in {model_dir}")

    run_record = json.loads(run_json_path.read_text())
    backbone = run_record["config"]["backbone"]
    thresholds = run_record.get("thresholds") or [0.5] * len(DEFAULT_LABELS)
    labels = DEFAULT_LABELS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiLabelClassifier(backbone, num_classes=len(labels)).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    _state["model"] = model
    _state["device"] = device
    _state["labels"] = labels
    _state["thresholds"] = thresholds
    _state["backbone"] = backbone
    _state["transform"] = eval_transforms()
    print(f"Loaded {backbone} from {ckpt_path} on {device}", flush=True)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if "model" not in _state:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthResponse(status="ok", backbone=_state["backbone"], num_labels=len(_state["labels"]))


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    if "model" not in _state:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Expected an image file, got {file.content_type}")

    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not decode image: {e}")

    tensor = _state["transform"](img).unsqueeze(0).to(_state["device"])
    with torch.no_grad():
        logits = _state["model"](tensor)
        probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()

    predictions = [
        LabelPrediction(
            label=name,
            probability=round(p, 4),
            threshold=_state["thresholds"][i],
            predicted=p >= _state["thresholds"][i],
        )
        for i, (name, p) in enumerate(zip(_state["labels"], probs))
    ]
    return PredictResponse(predictions=predictions, backbone=_state["backbone"])
