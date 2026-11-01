"""Track-level aggregation independent of the detector implementation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrackSummary:
    track_id: int
    class_id: int
    class_name: str
    first_frame: int
    last_frame: int
    observations: int = 0
    max_detection_confidence: float = 0.0
    severity_sums: dict[str, float] = field(default_factory=dict)
    severity_observations: int = 0
    last_severity_frame: int | None = None

    def update_detection(self, frame_index: int, confidence: float) -> None:
        self.last_frame = frame_index
        self.observations += 1
        self.max_detection_confidence = max(
            self.max_detection_confidence, confidence
        )

    def should_sample_severity(self, frame_index: int, interval: int) -> bool:
        return (
            self.last_severity_frame is None
            or frame_index - self.last_severity_frame >= interval
        )

    def update_severity(
        self, frame_index: int, probabilities: dict[str, float]
    ) -> None:
        for label, probability in probabilities.items():
            self.severity_sums[label] = (
                self.severity_sums.get(label, 0.0) + probability
            )
        self.severity_observations += 1
        self.last_severity_frame = frame_index

    def severity(self) -> tuple[str, float]:
        if not self.severity_observations:
            return "", 0.0
        averages = {
            label: total / self.severity_observations
            for label, total in self.severity_sums.items()
        }
        label = max(averages, key=averages.get)
        return label, averages[label]

    def as_dict(self, fps: float) -> dict:
        severity_label, severity_confidence = self.severity()
        return {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "first_time_seconds": self.first_frame / fps if fps else None,
            "last_time_seconds": self.last_frame / fps if fps else None,
            "observations": self.observations,
            "max_detection_confidence": self.max_detection_confidence,
            "severity": severity_label,
            "severity_confidence": severity_confidence,
            "severity_observations": self.severity_observations,
        }
