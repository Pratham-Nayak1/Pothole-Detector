# Experiment Log

Last updated: 2026-10-17

This document records the completed experiments represented by the local artifacts under `outputs/`. Generated weights, plots, crops, and reports are excluded from Git.

## Detection dataset

| Split | Images | Boxes | D00 | D10 | D20 | D40 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 25,677 | 64,293 | 27,114 | 16,358 | 9,941 | 10,880 |
| Validation | 4,509 | 11,470 | 4,807 | 2,992 | 1,769 | 1,902 |
| Total | 30,186 | 75,763 | 31,921 | 19,350 | 11,710 | 12,782 |

Dataset integrity checks found matching image/label counts, no empty label files, and no malformed YOLO rows in either split. Two duplicate D40 annotation rows—one in each split—were found and removed locally on 2026-10-17. Dataset files are excluded from Git.

## Detector training

| Run | Completed epochs | Best epoch | Precision | Recall | mAP50 | mAP50-95 | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| YOLO11n baseline, 640 px | 30 | 30 | 0.6682 | 0.5455 | 0.6040 | 0.3089 | Baseline completed |
| YOLO11s baseline, 640 px | 30 | 30 | 0.6979 | 0.5808 | 0.6470 | 0.3422 | Best run |
| YOLO11s low-LR fine-tune, 640 px | 13 | 1 | 0.6511 | 0.5684 | 0.6146 | 0.3157 | No improvement; early-stopped |
| YOLO11s fine-tune, 800 px | 11 | 1 | 0.6556 | 0.5619 | 0.6109 | 0.3078 | No improvement; early-stopped |

The YOLO11s 640 px baseline remains the selected detector.

The nominal low-LR fine-tune was later found to have used `optimizer=auto`; Ultralytics selected a substantially larger effective learning rate despite the requested `lr0=0.0001`. Its negative result is valid for the recorded run but does not test a pinned low-LR AdamW recipe. The corrected script now explicitly selects AdamW, disables warm-up, and writes to a new run directory.

## Threshold evaluation

Sixteen validation combinations were evaluated across confidence thresholds `0.05`, `0.10`, `0.15`, and `0.20`, and NMS IoU thresholds `0.40`, `0.50`, `0.60`, and `0.70`.

Best measured mAP50-95 configuration:

| Confidence | NMS IoU | Precision | Recall | mAP50 | mAP50-95 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 0.70 | 0.6984 | 0.5815 | 0.6267 | 0.3321 |

Selected operational inspection configuration:

| Confidence | NMS IoU | Precision | Recall | mAP50 | mAP50-95 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.20 | 0.40 | 0.6941 | 0.6098 | 0.5615 | 0.2988 |

On the 20-image NMS inspection sample, IoU `0.70` produced 43 boxes while IoU `0.50` and `0.40` each produced 38. On a separate 30-image inspection sample, the selected `0.20`/`0.40` configuration produced 69 detections, with two images receiving no detections.

Per-class results at the selected operational thresholds:

| Class | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| D00 Longitudinal Crack | 0.6532 | 0.5690 | 0.5119 | 0.2812 |
| D10 Transverse Crack | 0.6570 | 0.5646 | 0.5004 | 0.2538 |
| D20 Alligator Crack | 0.7038 | 0.6476 | 0.6016 | 0.3256 |
| D40 Pothole | 0.7624 | 0.6579 | 0.6322 | 0.3347 |

### Inference-quality comparison, 2026-10-17

Full-set validation was repeated with class-agnostic NMS. These runs evaluate candidate generation before the final class-specific reporting filter.

| Image size | Candidate confidence | NMS IoU | Agnostic | Precision | Recall | mAP50 | mAP50-95 | D40 mAP50-95 |
| ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| 640 | 0.10 | 0.40 | Yes | 0.7024 | 0.6036 | 0.5992 | 0.3124 | 0.3470 |
| 640 | 0.05 | 0.40 | Yes | 0.7024 | 0.6036 | 0.6184 | 0.3192 | 0.3540 |
| 640 | 0.05 | 0.70 | Yes | 0.7007 | 0.5803 | 0.6228 | 0.3304 | 0.3638 |
| 960 | 0.10 | 0.40 | Yes | 0.6506 | 0.5892 | 0.5613 | 0.2636 | 0.3260 |

The 640 px / 0.05 / 0.70 candidate setting was selected. Candidate NMS is deliberately permissive for mAP and recall; final output is deduplicated at stricter overlap thresholds. The 960 px test was rejected because it reduced every aggregate metric while more than doubling inference time (approximately 4.3 ms to 10.2 ms per image on the test GPU).

Final operational filtering keeps D00/D10/D20 at confidence `0.20`, lowers only D40 to `0.10`, and removes cross-class duplicates at IoU `0.45` or intersection-over-smaller-box `0.80`. On the known difficult Norway image, 32 permissive raw candidates were reduced to 8 reported detections. The result recovered the three labelled D40 regions and also retained one extra D40 candidate at confidence `0.128`; this illustrates the intended recall/precision tradeoff and the need for a reviewed acceptance set.

## Severity dataset

The completed review batch contains 327 detections from 100 source images.

| Label | Count |
| --- | ---: |
| Low | 108 |
| Medium | 148 |
| High | 54 |
| Duplicate/excluded | 17 |
| Usable total | 310 |

The classifier split is grouped by source image so crops from one source image cannot appear in multiple train, validation, or test partitions.

Fixed split manifest generated on 2026-10-17:

| Split | Examples | Source images | Low | Medium | High |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 202 | 64 | 68 | 101 | 33 |
| Validation | 40 | 16 | 19 | 15 | 6 |
| Test | 68 | 20 | 21 | 32 | 15 |

## Severity classifier

Architecture: ImageNet-pretrained ResNet18 with a three-class output layer.

- Best epoch: 4 of 15
- Best validation loss: 0.8084
- Held-out test examples: 68
- Test accuracy: 0.6176
- Test balanced accuracy: 0.6038

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Low | 0.5238 | 0.5238 | 0.5238 | 21 |
| Medium | 0.6875 | 0.6875 | 0.6875 | 32 |
| High | 0.6000 | 0.6000 | 0.6000 | 15 |

The result establishes a baseline but is not sufficient for production use because the labelled dataset and test set are small.

### Ordinal-aware ResNet18 experiment

On 2026-10-17, the same fixed 202/40/68 split was used to test weighted cross-entropy plus a distance-aware ordinal penalty. Model selection used validation balanced accuracy with early stopping.

- Best epoch: 2 of 15; stopped after epoch 7.
- Test accuracy: `0.6029`.
- Test balanced accuracy: `0.5770`.
- Macro F1: `0.5730`.
- Ordinal mean absolute error: `0.5000` severity levels.
- Quadratic weighted kappa: `0.2941`.
- Within-one-level accuracy: `0.8971`.

The experiment did not beat the deployed flat-classification baseline and was not promoted. At confidence `0.60`, selective accuracy reached `0.7826`, but coverage was only `0.3382`; this supports an eventual `uncertain` outcome but does not compensate for the small dataset.

## Expanded severity review batch

The 2026-10-17 review batch was generated with the deployed D40 thresholds and post-processing:

- Source images selected: 250.
- Validation images inspected to find them: 1,731.
- Candidate crops requiring review: 772.
- Candidate confidence threshold: `0.05`; D40 final threshold: `0.10`.
- Reviewer-1 labels complete: 0 at generation time.

The batch is intentionally unlabelled in Git and stored under ignored `outputs/`. It must not enter training until human review resolves false positives, uncertain examples, duplicates, and reviewer disagreements.

## End-to-end pipeline smoke test

On 2026-10-17, the unified image pipeline was tested with both trained checkpoints on `rdd_Norway_Norway_000440.jpg` using confidence `0.20` and NMS IoU `0.40`.

- Images processed: 1
- Total detections: 5
- Pothole detections: 1
- Predicted pothole severity: high
- Severity confidence: 0.9893
- Expected manual severity label from the review dataset: high
- Outputs verified: annotated image, CSV report, and schema-versioned JSON report

This is a functional smoke test, not an independent accuracy measurement.

The smoke test was repeated after the inference-quality changes. It produced 32 raw candidates, 8 final detections, and 4 D40 candidates. All three ground-truth pothole regions were represented; one additional low-confidence D40 candidate remained. Annotation output now contains only post-processed boxes with consolidated detection and severity labels.

## Video pipeline smoke tests

On 2026-10-17, ByteTrack video inference was tested with a short clip derived from the same known pothole image.

Native-resolution test (`3650 × 2044`, 10 frames):

- Tracks: 3
- Unique pothole tracks: 1
- Pothole track observations: 10
- Aggregated severity: high
- Severity confidence: 0.9830
- Untracked detections: 0

A separate 720p test tracked one crack but did not retain the pothole at confidence `0.20`. The original still-image pothole confidence was only approximately `0.225`, indicating that input resolution materially affects this edge case. Real-video validation should therefore compare native-resolution, tiled, and lower-resolution inference before selecting a production policy.
