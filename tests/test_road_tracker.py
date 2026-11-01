"""Unit tests for the dedicated RoadTracker."""

from src.detection.postprocess import Detection
from src.video.tracker import RoadTracker


def test_road_tracker_assigns_ids_to_new_detections() -> None:
    tracker = RoadTracker()
    d1 = Detection(class_id=3, confidence=0.8, xyxy=(100, 100, 200, 200))
    d2 = Detection(class_id=0, confidence=0.7, xyxy=(300, 300, 350, 400))

    tracked = tracker.update([d1, d2], frame_index=0)

    assert len(tracked) == 2
    assert tracked[0].track_id == 1
    assert tracked[1].track_id == 2


def test_road_tracker_maintains_track_across_frames() -> None:
    tracker = RoadTracker(iou_threshold=0.15)
    d1 = Detection(class_id=3, confidence=0.8, xyxy=(100, 100, 200, 200))
    tracker.update([d1], frame_index=0)

    # Frame 1: Pothole moves slightly downward due to vehicle motion
    d1_next = Detection(class_id=3, confidence=0.85, xyxy=(105, 120, 205, 220))
    tracked_next = tracker.update([d1_next], frame_index=1)

    assert len(tracked_next) == 1
    assert tracked_next[0].track_id == 1
    assert tracker.active_tracks[1].hits == 2


def test_road_tracker_prunes_expired_tracks() -> None:
    tracker = RoadTracker(max_age_frames=5)
    d1 = Detection(class_id=3, confidence=0.8, xyxy=(100, 100, 200, 200))
    tracker.update([d1], frame_index=0)

    assert 1 in tracker.active_tracks

    # 10 frames pass without detection
    tracker.update([], frame_index=10)

    assert 1 not in tracker.active_tracks
