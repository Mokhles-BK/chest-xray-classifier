"""Pydantic request/response models for the inference API."""
from __future__ import annotations

from pydantic import BaseModel


class LabelPrediction(BaseModel):
    label: str
    probability: float
    threshold: float
    predicted: bool


class PredictResponse(BaseModel):
    predictions: list[LabelPrediction]
    backbone: str


class HealthResponse(BaseModel):
    status: str
    backbone: str
    num_labels: int
