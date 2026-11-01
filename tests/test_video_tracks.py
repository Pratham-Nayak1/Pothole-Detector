from src.video.tracks import TrackSummary


def test_track_summary_aggregates_detection_and_severity() -> None:
    track = TrackSummary(
        track_id=7,
        class_id=3,
        class_name="Pothole (D40)",
        first_frame=10,
        last_frame=10,
    )
    track.update_detection(10, 0.6)
    track.update_detection(11, 0.8)
    track.update_severity(10, {"low": 0.1, "medium": 0.2, "high": 0.7})
    track.update_severity(20, {"low": 0.2, "medium": 0.3, "high": 0.5})

    row = track.as_dict(fps=10.0)

    assert row["first_time_seconds"] == 1.0
    assert row["last_time_seconds"] == 1.1
    assert row["observations"] == 2
    assert row["max_detection_confidence"] == 0.8
    assert row["severity"] == "high"
    assert row["severity_confidence"] == 0.6


def test_track_summary_controls_severity_sampling_interval() -> None:
    track = TrackSummary(1, 3, "Pothole", 0, 0)

    assert track.should_sample_severity(0, interval=10)
    track.update_severity(0, {"low": 1.0})
    assert not track.should_sample_severity(9, interval=10)
    assert track.should_sample_severity(10, interval=10)
