# Results — ChestX-ray14 Multi-Label Classifier

## Setup
- Backbones compared: ResNet18 and EfficientNet-B0 (both ImageNet-pretrained,
  linear head, sigmoid + BCE)
- Data: 77,967 train / 8,557 validation images, 15 labels (multi-label)
- Batch size 16, AdamW, cosine LR schedule, early stopping (patience=3)
- Trained locally on an RTX 3050 (4GB VRAM)
- Served via a FastAPI inference endpoint, containerized with Docker (see
  `api/` and `Dockerfile`)

## Headline result
Multi-label classification is not evaluated with accuracy — a model can score
90%+ accuracy by predicting "no finding" on everything, which is why every
number below is precision/recall/F1, matching standard practice for this task.

| Model             | Params | Epoch time (3050) | Macro F1 @0.5 | Macro F1 @tuned | Weighted F1 @tuned |
|--------------------|-------:|-------------------:|--------------:|----------------:|-------------------:|
| ResNet18           | 11.2M  | ~355-410s          | 0.165         | **0.264**       | **0.533**          |
| EfficientNet-B0    | 4.0M   | ~510-530s          | 0.198         | 0.261           | 0.523              |

Per-class threshold tuning (maximizing F1 per class on the validation set,
rather than a blanket 0.5 cutoff) recovered a large share of performance on
rare classes that a fixed threshold effectively silences, for both models.

**Architecture comparison finding:** the two backbones land within noise of
each other on final tuned performance (0.264 vs 0.261 macro F1) — not a
meaningful difference either way. EfficientNet-B0 starts from a stronger
position at a fixed 0.5 threshold (0.198 vs 0.165) and uses under half the
parameters (4.0M vs 11.2M), but trains slower per epoch on this GPU (likely
due to depthwise-separable convolutions parallelizing less efficiently than
ResNet's simpler blocks on this hardware). Reported as an honest tie with a
real resource tradeoff, not a case for one model being "better."

## Per-class performance, tuned thresholds (ResNet18)

| Label              | Precision | Recall | F1    | Support |
|---------------------|----------:|-------:|------:|--------:|
| No Finding          | 0.709     | 0.775  | 0.775 | 4956    |
| Effusion            | 0.486     | -      | 0.554 | 868     |
| Infiltration        | 0.294     | -      | 0.332 | 1409    |
| Mass                | 0.424     | -      | 0.360 | 393     |
| Atelectasis         | 0.339     | -      | 0.315 | 806     |
| Cardiomegaly        | 0.401     | -      | 0.282 | 162     |
| Emphysema           | 0.430     | -      | 0.318 | 173     |
| Pneumothorax        | 0.324     | -      | 0.245 | 257     |
| Nodule              | 0.263     | -      | 0.203 | 473     |
| Edema               | 0.212     | -      | 0.180 | 137     |
| Pleural_Thickening  | 0.176     | -      | 0.109 | 246     |
| Consolidation       | 0.148     | -      | 0.129 | 260     |
| Hernia              | 0.273     | -      | 0.086 | 19      |
| Fibrosis            | 0.127     | -      | 0.042 | 126     |
| Pneumonia           | 0.078     | -      | 0.029 | 98      |

## Per-class performance, tuned thresholds (EfficientNet-B0)

| Label              | Precision | Recall | F1    | Support |
|---------------------|----------:|-------:|------:|--------:|
| No Finding          | 0.682     | -      | 0.768 | 4956    |
| Effusion            | 0.460     | -      | 0.519 | 868     |
| Infiltration        | 0.269     | -      | 0.312 | 1409    |
| Mass                | 0.360     | -      | 0.351 | 393     |
| Atelectasis         | 0.326     | -      | 0.330 | 806     |
| Emphysema           | 0.450     | -      | 0.307 | 173     |
| Pneumothorax        | 0.340     | -      | 0.280 | 257     |
| Cardiomegaly        | 0.295     | -      | 0.208 | 162     |
| Hernia              | 0.750     | -      | 0.237 | 19      |
| Edema               | 0.236     | -      | 0.172 | 137     |
| Nodule              | 0.277     | -      | 0.169 | 473     |
| Pleural_Thickening  | 0.266     | -      | 0.117 | 246     |
| Consolidation       | 0.128     | -      | 0.086 | 260     |
| Fibrosis            | 0.140     | -      | 0.038 | 126     |
| Pneumonia           | 0.083     | -      | 0.026 | 98      |

(Recall column omitted where not recorded verbatim — see each run's
`run.json` under `runs/` for the complete record: `runs/resnet18_20260923_023010_49ec6d/`
and `runs/efficientnet_b0_baseline/`.)

Both models show the same pattern: the weakest classes (Hernia, Pneumonia,
Fibrosis, Consolidation) all have the lowest support in the training set
(19-260 positive examples out of ~78k) — consistent with class imbalance
being the primary limiting factor across architectures, not a property of
either specific model.

## Ablation: class-weighted loss (pos_weight)

Tested upweighting rare-class positives directly in the BCE loss (neg/pos
ratio per class, capped at 20x) on ResNet18, to address imbalance during
training rather than only at inference.

| Approach                          | Macro P | Macro R | Macro F1 |
|-------------------------------------|--------:|--------:|---------:|
| Baseline + tuned thresholds         | 0.446   | 0.163   | 0.264    |
| pos_weight + tuned thresholds       | 0.303   | 0.440   | 0.254    |

**Finding:** pos_weight substantially increased recall (catches more true
positives) at a proportional cost to precision (more false positives),
resulting in a near-identical overall F1. For this backbone and dataset,
per-class threshold tuning was the more effective lever — reweighting the
loss did not provide a net improvement over it. Documented here as a tested
approach with a negative result, rather than discarded.

## Deployment

The ResNet18 checkpoint is served via a FastAPI `/predict` endpoint
(`api/app.py`), returning per-class probabilities and predictions using the
tuned per-class thresholds saved alongside the model. Containerized with
Docker (CPU-only inference — no GPU passthrough needed for serving); model
weights are mounted at runtime, never baked into the image or committed to
git. Verified working end-to-end in the container.

## Known limitations / next steps
- Rare classes (Hernia, Pneumonia, Fibrosis) remain weak even after tuning,
  across both architectures tested; likely needs either more aggressive
  resampling, a different imbalance strategy (e.g. focal loss), or more data
  for those specific classes to improve meaningfully.
- Only one ablation (pos_weight) was tested; other imbalance strategies
  (focal loss, oversampling rare classes) were not explored.
- The served API currently exposes the ResNet18 checkpoint; swapping in the
  EfficientNet-B0 checkpoint would only require pointing `MODEL_DIR` at its
  run directory, no code change.
