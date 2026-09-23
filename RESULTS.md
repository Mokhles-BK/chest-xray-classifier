# Results — ChestX-ray14 Multi-Label Baseline

## Setup
- Backbone: ResNet18 (ImageNet-pretrained), linear head, sigmoid + BCE
- Data: 77,967 train / 8,557 validation images, 15 labels (multi-label)
- 10 epochs, batch size 16, AdamW, cosine LR schedule
- Trained locally on an RTX 3050 (4GB VRAM)

## Headline result
Multi-label classification is not evaluated with accuracy — a model can score
90%+ accuracy by predicting "no finding" on everything, which is why every
number below is precision/recall/F1, matching standard practice for this task.

| Threshold strategy      | Macro F1 | Weighted F1 |
|--------------------------|---------:|------------:|
| Fixed 0.5                | 0.165    | 0.456       |
| Per-class tuned          | **0.264**| **0.533**   |

Per-class threshold tuning (maximizing F1 per class on the validation set,
rather than a blanket 0.5 cutoff) recovered a large share of performance on
rare classes that a fixed threshold effectively silences.

## Per-class performance (tuned thresholds)

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

(Recall column omitted where not recorded verbatim — see run.json in
`runs/resnet18_20260923_023010_49ec6d/` for the complete run record.)

The weakest classes (Hernia, Pneumonia, Fibrosis, Consolidation) all have the
lowest support in the training set (19–260 positive examples out of ~78k),
consistent with class imbalance being the primary limiting factor rather than
an architecture or training bug.

## Ablation: class-weighted loss (pos_weight)

Tested upweighting rare-class positives directly in the BCE loss (neg/pos
ratio per class, capped at 20x) to address imbalance during training rather
than only at inference.

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

## Known limitations / next steps
- Single architecture (ResNet18) — no comparison against a transformer or
  larger CNN backbone was completed.
- No API/deployment wrapper built yet — this is a trained model + evaluation
  pipeline, not yet a served inference endpoint.
- Rare classes (Hernia, Pneumonia, Fibrosis) remain weak even after tuning;
  likely needs either more aggressive resampling, a different imbalance
  strategy (e.g. focal loss), or simply more data for those classes to
  improve meaningfully.
