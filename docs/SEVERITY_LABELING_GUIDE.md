# Pothole Severity Labelling Guide

Last updated: 2026-10-17

This guide defines the visual labels used by the prototype severity classifier. These labels describe visible surface impact; they are not engineering measurements of depth, structural damage, or repair cost.

## Labelling principles

- Judge the highlighted pothole, not the overall condition of the road.
- Use the full source image as context while checking the saved crop.
- Do not infer depth when shadows, water, glare, or camera angle make it unknowable.
- Mark ambiguous examples as `uncertain`; do not force them into a severity class.
- Mark repeated detections of the same pothole in one source image as `duplicate`.
- Exclude detections that are not potholes.

## Severity classes

### Low

- Small visible footprint relative to the lane.
- Limited surface breakup.
- Shallow appearance with little shadow or exposed subsurface material.
- Likely to require monitoring or minor repair.

### Medium

- Clear pothole with a moderate footprint or visible material loss.
- Noticeable edge breakdown, shadow, or exposed underlying layer.
- Large enough to affect a normal vehicle path but not visually extreme.

### High

- Large footprint, pronounced breakup, or strong evidence of depth.
- Jagged edges, substantial material loss, multiple connected cavities, or exposed base.
- Appears likely to pose an immediate hazard or require priority repair.

## Non-severity outcomes

- `uncertain`: visibility or context is inadequate for a defensible label.
- `duplicate`: another row represents the same pothole in the same source image.
- `not_pothole`: detection is a crack, patch, shadow, drain, puddle, or other false positive.

The collection tool records `not_pothole` with keyboard key `6`.

## Review quality process

For the next batch:

1. Label at least 1,000 usable examples with deliberate sampling across data sources.
2. Have a second reviewer independently label at least 20% of the examples.
3. Measure agreement before resolving disagreements.
4. Keep the original reviewer labels as separate columns rather than overwriting them.
5. Resolve disagreements through review and record the adjudicated label.
6. Keep all crops from a source image in one train, validation, or test split.

## Current 250-image review batch

The reproducible batch generated on 2026-10-17 contains 250 source images and 772 D40 candidates at `outputs/severity_review_250`. It deliberately includes borderline predictions and false positives; reviewers must use `not_pothole` instead of assigning severity to those cases.

Primary review:

```powershell
python -m streamlit run src/severity/review_app.py -- --batch outputs/severity_review_250 --reviewer reviewer_1
```

Independent 20% review, sampled within the first reviewer's labels:

```powershell
python -m streamlit run src/severity/review_app.py -- --batch outputs/severity_review_250 --reviewer reviewer_2 --sample-percent 20
```

Resolve only disagreements after both reviews:

```powershell
python -m streamlit run src/severity/review_app.py -- --batch outputs/severity_review_250 --reviewer adjudicator
```

The interface highlights one candidate in the full source image, shows its padded crop, hides reviewer 1's decision from reviewer 2, and saves after every label. Reviewer labels are never overwritten during adjudication.

## Known limitation

Visual severity is only a proxy. Reliable physical severity may require calibrated camera geometry, scale references, depth sensors, or multiple viewpoints.
