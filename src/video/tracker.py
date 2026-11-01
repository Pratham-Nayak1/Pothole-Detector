"""Spatial road defect tracker designed for forward-facing vehicle cameras.

Unlike standard ByteTrack which hides new detections until multi-frame Kalman
confirmation (causing road defects to remain invisible until they are right in front
of the camera), RoadTracker tracks and displays road defects immediately from their
very first appearance ('at a prior basis').

It associates detections across frames using IoU overlap and center-distance
proximity to account for forward road perspective motion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from src.detection.postprocess import Detection, overlap_scores


@dataclass
class ActiveTrack:
    track_id: int
    class_id: int
    xyxy: tuple[int, int, int, int]
    last_frame: int
    start_frame: int
    hits: int = 1


class RoadTracker:
    """Multi-object tracker tuned for road defect inspection videos."""

    def __init__(
        self,
        *,
        iou_threshold: float = 0.15,
        max_center_distance: float = 140.0,
        max_age_frames: int = 30,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_center_distance = max_center_distance
        self.max_age_frames = max_age_frames
        self.next_track_id = 1
        self.active_tracks: dict[int, ActiveTrack] = {}

    def update(
        self,
        detections: Sequence[Detection],
        frame_index: int,
    ) -> list[Detection]:
        """Associate detections with existing tracks or assign new track IDs.

        Every input detection is guaranteed to receive a valid track_id.
        """
        if not detections:
            # Clean up expired tracks
            self._prune(frame_index)
            return []

        # Remove tracks older than max_age_frames
        self._prune(frame_index)

        assigned_detections: list[Detection] = []
        unmatched_indices = list(range(len(detections)))

        # 1. Match existing active tracks to incoming detections of the same class
        for track_id, track in list(self.active_tracks.items()):
            prev_box = track.xyxy
            t_det = Detection(track.class_id, 1.0, prev_box)
            tx1, ty1, tx2, ty2 = prev_box
            tcx = (tx1 + tx2) / 2.0
            tcy = (ty1 + ty2) / 2.0

            match_candidates = []
            for idx in unmatched_indices:
                det = detections[idx]
                if det.class_id != track.class_id:
                    continue

                iou, ios = overlap_scores(det, t_det)
                dx1, dy1, dx2, dy2 = det.xyxy
                dcx = (dx1 + dx2) / 2.0
                dcy = (dy1 + dy2) / 2.0
                dist = math.hypot(tcx - dcx, tcy - dcy)

                # Score matches: prefer IoU overlap, sub-box containment, or center proximity
                if iou >= self.iou_threshold or ios >= 0.10 or dist <= self.max_center_distance:
                    match_candidates.append((idx, iou, ios, dist))

            if match_candidates:
                # A track may consume only one detection in a frame. Prefer overlap,
                # then containment, then nearest center.
                matched_indices = [
                    max(match_candidates, key=lambda item: (item[1], item[2], -item[3]))[0]
                ]
                matched_dets = [detections[i] for i in matched_indices]
                ux1 = min(d.xyxy[0] for d in matched_dets)
                uy1 = min(d.xyxy[1] for d in matched_dets)
                ux2 = max(d.xyxy[2] for d in matched_dets)
                uy2 = max(d.xyxy[3] for d in matched_dets)

                track.xyxy = (ux1, uy1, ux2, uy2)
                track.last_frame = frame_index
                track.hits += len(matched_dets)
                best_conf = max(d.confidence for d in matched_dets)

                assigned_detections.append(
                    Detection(
                        class_id=track.class_id,
                        confidence=best_conf,
                        xyxy=track.xyxy,
                        track_id=track_id,
                    )
                )
                for idx in matched_indices:
                    unmatched_indices.remove(idx)

        # 2. Check if any remaining unmatched detections overlap or lie inside an active track
        still_unmatched = []
        for idx in unmatched_indices:
            det = detections[idx]
            absorbed = False
            for track_id, track in self.active_tracks.items():
                if det.class_id != track.class_id:
                    continue
                iou, ios = overlap_scores(det, Detection(track.class_id, 1.0, track.xyxy))
                if iou > 0.0 or ios >= 0.10:
                    tx1, ty1, tx2, ty2 = track.xyxy
                    dx1, dy1, dx2, dy2 = det.xyxy
                    track.xyxy = (min(tx1, dx1), min(ty1, dy1), max(tx2, dx2), max(ty2, dy2))
                    track.last_frame = frame_index
                    track.hits += 1
                    assigned_detections.append(
                        Detection(
                            class_id=det.class_id,
                            confidence=det.confidence,
                            xyxy=track.xyxy,
                            track_id=track_id,
                        )
                    )
                    absorbed = True
                    break
            if not absorbed:
                still_unmatched.append(idx)

        # 3. Allocate new track IDs for genuinely new unmatched detections
        for idx in still_unmatched:
            det = detections[idx]
            new_id = self.next_track_id
            self.next_track_id += 1
            self.active_tracks[new_id] = ActiveTrack(
                track_id=new_id,
                class_id=det.class_id,
                xyxy=det.xyxy,
                last_frame=frame_index,
                start_frame=frame_index,
                hits=1,
            )
            assigned_detections.append(
                Detection(
                    class_id=det.class_id,
                    confidence=det.confidence,
                    xyxy=det.xyxy,
                    track_id=new_id,
                )
            )

        return assigned_detections

    def _prune(self, frame_index: int) -> None:
        expired = [
            tid
            for tid, t in self.active_tracks.items()
            if frame_index - t.last_frame > self.max_age_frames
        ]
        for tid in expired:
            del self.active_tracks[tid]
