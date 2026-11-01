# Project Status

Last updated: 2026-10-17

## Summary

The project provides offline image and video road-defect inference, clip-level tracking, and prototype pothole-severity classification. The YOLO11s detector now uses a recall-oriented candidate threshold, class-specific final thresholds, and class-agnostic duplicate suppression. Severity classification remains data-limited; GPS-backed cross-journey identity, analytics, and a user-facing dashboard have not yet been implemented.

## Component status

| Component | Status | Notes |
| --- | --- | --- |
| Four-class YOLO dataset | Complete | 30,186 images and 75,763 validated boxes across train and validation splits. |
| YOLO11n baseline | Complete | Trained for 30 epochs. |
| YOLO11s 640 baseline | Complete | Best detector so far. |
| YOLO11s fine-tuning experiments | Complete | Low-learning-rate and 800 px runs did not improve the baseline. |
| Threshold evaluation | Complete | Confidence/NMS grid and visual comparisons generated. |
| Annotation review tooling | Complete | Ground-truth and prediction comparisons generated for sample images. |
| Manual severity labelling | Complete for initial batch | 327 detections reviewed from 100 images. |
| Expanded severity review | Batch prepared; human review pending | 250 source images and 772 post-processed D40 candidates in a resumable two-reviewer workflow. |
| Severity dataset preparation | Complete | 310 usable examples after removing 17 duplicates. |
| Fixed severity splits | Complete | Deterministic 202/40/68 train/validation/test examples with source-image isolation. |
| ResNet18 severity baseline | Complete | Initial three-class model trained and evaluated. |
| End-to-end image inference | Complete | One CLI processes an image or directory through detection and optional D40 severity classification. |
| Video ingestion and processing | Clip phase complete | Video frames are detected, annotated, and exported with track reports. |
| Multi-frame tracking | Clip phase complete | ByteTrack associates defects inside one continuous clip. |
| Unique-defect deduplication | Partial | Track IDs prevent adjacent-frame recounting; GPS/cross-journey deduplication is not implemented. |
| Road-maintenance analytics | Not started | `src/analytics` is reserved for this work. |
| Streamlit dashboard | Not started | Streamlit is installed but no application exists. |
| Exportable inspection reports | Image/video phase complete | Both pipelines export annotated media plus versioned CSV/JSON reports. PDF and route-level reports remain planned. |
| Shared pipeline configuration | Complete | Model paths, thresholds, image sizes, output settings, and device selection are centralized. |
| Dataset integrity audit | Complete | Validates pairs, rows, ranges, classes, and duplicate annotations; optional cross-split hashing is available. |
| Per-class detector evaluation | Complete | Exports aggregate and per-class CSV/JSON metrics. |
| Duplicate-box suppression | Complete for single frames | Class-agnostic NMS plus IoU/containment filtering is shared by image and video output. |
| Missed-pothole mitigation | Initial pass complete | Candidates start at 0.05 and D40 is retained from 0.10; real-route acceptance testing remains required. |
| Automated tests | Severity-review phase complete | Twenty-two tests cover configuration, dataset auditing, post-processing, review resolution, dataset merging, image I/O, reports, severity splits, and track aggregation. |

## Current model choices

The downstream detection and severity-labelling scripts use:

- Checkpoint: `outputs/training/yolo11s_640_baseline/weights/best.pt`
- Input size: 640 px
- Candidate confidence threshold: 0.05
- Final confidence threshold: 0.20 for cracks and 0.10 for D40 potholes
- Candidate NMS IoU threshold: 0.70
- Class-agnostic NMS: enabled
- Secondary duplicate rules: IoU 0.45 or intersection-over-smaller-box 0.80

The selected candidate-generation setting (`0.05`, IoU `0.70`, class-agnostic NMS) measured aggregate mAP50-95 `0.3304`, compared with `0.2988` for the previous operational `0.20`/`0.40` setting. D40 mAP50-95 improved from `0.3347` to `0.3638`. The checkpoint's standard training-time validation score remains mAP50-95 `0.3422`; post-processing improves operational behavior but does not replace retraining.

A 960 px validation run was rejected: aggregate mAP50-95 fell to `0.2636` because this checkpoint was trained at 640 px. The inference size therefore remains 640 rather than assuming that a larger input is automatically better.

## Known limitations

- The dataset source, licence information, transformation history, and split-generation procedure are not yet documented.
- The detector configuration contains a machine-specific local data path.
- Legacy standalone training scripts explicitly select GPU device `0`; the image, video, audit, and evaluation CLIs support automatic CPU/GPU selection.
- The severity dataset has only 310 usable crops and is class-imbalanced.
- Severity labels reflect visual judgement and have no recorded multi-reviewer agreement process.
- Severity predictions are visual estimates and should not yet drive maintenance decisions without human review.
- Model artifacts and datasets are local-only and cannot be reconstructed from the repository.
- Dependency versions are not yet locked to exact releases.

## Phase completion

### Phase 1 — Evaluation foundation (complete, 2026-10-17)

- Centralized pipeline configuration and automatic CPU/GPU selection.
- Dataset-integrity audit with duplicate-row detection and optional exact cross-split hashing.
- Aggregate and per-class detector evaluation exports.
- Initial automated test suite.

### Phase 2 — Unified image inference (complete, 2026-10-17)

- Single-image and directory input, including optional recursive discovery.
- YOLO defect detection and ResNet18 severity classification for D40 crops.
- Resolution-aware annotated image output.
- CSV and schema-versioned JSON reports with model paths and thresholds.
- Verified against a real validation image and both trained checkpoints.

### Phase 3 — Severity quality (in progress, 2026-10-17)

- Added a written low/medium/high labelling standard and review-quality process.
- Added explicit `uncertain`, `duplicate`, and `not_pothole` review outcomes.
- Added reusable severity-dataset validation and deterministic group-isolated splits.
- Updated training to save the exact train, validation, and test manifests.
- Data expansion and independent second-reviewer labels remain outstanding.

### Phase 4 — Video tracking (clip-level complete, 2026-10-17)

- Added video ingestion and annotated MP4 output.
- Added ByteTrack association with persistent defect IDs inside a clip.
- Added periodic severity sampling and probability aggregation per pothole track.
- Added track-level CSV and schema-versioned JSON reports.
- Verified on a native-resolution smoke clip: 10 frames, 3 tracks, 1 pothole track, high severity at 0.983 confidence.
- Confirmed that downscaling the same source to 720p can suppress a low-confidence pothole, so resolution policy requires validation on real videos.

### Phase 5 — Detection quality and duplicate control (initial pass complete, 2026-10-17)

- Added configurable candidate and per-class final confidence thresholds.
- Lowered only the D40 final threshold to recover borderline potholes without lowering crack thresholds.
- Enabled class-agnostic NMS to suppress boxes that describe the same damage with different predicted classes.
- Added a second class-agnostic IoU/containment filter shared by still-image and tracked-video output.
- Replaced raw Ultralytics annotations with consolidated labels drawn only for retained detections.
- Added evaluator overrides and full-validation comparisons at 640 and 960 px.
- Corrected the low-learning-rate fine-tuning recipe by pinning AdamW; the earlier `optimizer=auto` run did not honor the intended effective learning rate.
- Verified the difficult Norway sample: all three labelled pothole regions were recovered, with one additional low-confidence D40 candidate requiring acceptance-set review.
- Re-verified native-resolution video: 10 frames, 3 stable tracks, 1 pothole track, and no untracked observations.

### Phase 6 — Expanded severity learning (review stage, 2026-10-17)

- Replaced the hard-coded, non-resumable collector with a configurable batch generator that checkpoints progress safely.
- Generated a local batch of 250 source images containing 772 D40 candidates, crops, and contextual annotations.
- Added a Streamlit review interface for primary review, sampled blind second review, and disagreement adjudication.
- Added explicit label provenance and multi-batch merging without silently resolving reviewer disagreement.
- Improved deterministic source-image-isolated splitting by searching for closer class balance.
- Added an ordinal-aware loss, early stopping, macro/ordinal metrics, and confidence-coverage reporting.
- The ordinal ResNet18 experiment on the original 310 examples was not promoted: accuracy `0.6029` and balanced accuracy `0.5770` were below the deployed `0.6176`/`0.6038` baseline.
- Human review of the new batch is the remaining prerequisite before training the expanded model.

## Recommended next milestones

1. Complete primary review of the 772-candidate severity batch, independently review 20%, and adjudicate disagreements.
2. Retrain severity on the merged dataset and promote only if macro F1, balanced accuracy, and ordinal error improve on the fixed test set.
3. Build a fixed, manually reviewed detection acceptance set containing misses, duplicates, small potholes, shadows, patches, and puddles.
4. Run the corrected AdamW detector fine-tune and promote only if it beats `0.3422` mAP50-95 and improves D40 recall on the acceptance set.
5. Document dataset provenance and provide a reproducible dataset preparation/splitting script.
6. Validate resolution, throughput, and track stability on real inspection footage.
7. Implement GPS-backed cross-journey deduplication, analytics, reporting, and the operational dashboard.
