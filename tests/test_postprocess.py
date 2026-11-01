from src.detection.postprocess import Detection, filter_detections, overlap_scores


def detection(class_id: int, confidence: float, xyxy=(0, 0, 100, 100)) -> Detection:
    return Detection(class_id=class_id, confidence=confidence, xyxy=xyxy)


def test_overlap_scores_detect_contained_duplicate() -> None:
    large = detection(0, 0.8, (0, 0, 100, 100))
    contained = detection(3, 0.7, (20, 20, 80, 80))

    iou, ios = overlap_scores(large, contained)

    assert iou == 0.36
    assert ios == 1.0


def test_filter_uses_lower_pothole_threshold_and_removes_cross_class_duplicate() -> None:
    strongest = detection(3, 0.26, (10, 10, 90, 90))
    duplicate_other_class = detection(2, 0.21, (12, 12, 88, 88))
    weak_crack = detection(0, 0.15, (200, 200, 250, 250))
    separate_crack = detection(1, 0.25, (300, 300, 350, 350))
    borderline_pothole = detection(3, 0.12, (400, 400, 450, 450))

    filtered = filter_detections(
        [
            duplicate_other_class,
            weak_crack,
            separate_crack,
            strongest,
            borderline_pothole,
        ],
        default_confidence=0.2,
        class_confidence={3: 0.1},
        duplicate_iou=0.45,
        duplicate_ios=0.8,
    )

    assert filtered == [strongest, separate_crack, borderline_pothole]


def test_filter_keeps_nearby_non_overlapping_damage() -> None:
    first = detection(0, 0.8, (0, 0, 100, 20))
    second = detection(0, 0.7, (80, 0, 180, 20))

    filtered = filter_detections(
        [first, second],
        default_confidence=0.2,
        class_confidence={},
        duplicate_iou=0.45,
        duplicate_ios=0.8,
    )

    assert filtered == [first, second]
