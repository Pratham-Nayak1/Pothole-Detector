"""Spatial and velocity-guided road defect tracker for forward-facing vehicle cameras.

Unlike standard ByteTrack which hides new detections until multi-frame Kalman
confirmation (causing road defects to remain invisible until they are right in front
of the camera), RoadTracker tracks and displays road defects immediately from their
very first appearance ('at a prior basis').

Associates detections across frames using one-to-one optimal matching with velocity
prediction, IoU overlap, scale-consistent gating, and center-distance proximity.
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
    vx: float = 0.0
    vy: float = 0.0

    def predicted_xyxy(self, frame_index: int) -> tuple[int, int, int, int]:
        dt = max(0, min(8, frame_index - self.last_frame))
        if dt == 0 or (self.vx == 0.0 and self.vy == 0.0):
            return self.xyxy
        dx = int(round(self.vx * dt))
        dy = int(round(self.vy * dt))
        x1, y1, x2, y2 = self.xyxy
        return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


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
        """Associate detections with existing tracks one-to-one or assign new track IDs.

        Every input detection is guaranteed to receive a unique, valid track_id.
        """
        if not detections:
            self._prune(frame_index)
            return []

        self._prune(frame_index)

        # Build candidate (score, track_id, det_index) pairs for one-to-one matching
        candidate_pairs: list[tuple[float, int, int]] = []

        for track_id, track in self.active_tracks.items():
            tx1, ty1, tx2, ty2 = track.xyxy
            tw, th = max(1, tx2 - tx1), max(1, ty2 - ty1)
            t_area = float(tw * th)

            pred_box = track.predicted_xyxy(frame_index)
            px1, py1, px2, py2 = pred_box
            pcx = (px1 + px2) / 2.0
            pcy = (py1 + py2) / 2.0
            tcx = (tx1 + tx2) / 2.0
            tcy = (ty1 + ty2) / 2.0

            t_curr_det = Detection(track.class_id, 1.0, track.xyxy)
            t_pred_det = Detection(track.class_id, 1.0, pred_box)

            for det_idx, det in enumerate(detections):
                if det.class_id != track.class_id:
                    continue

                dx1, dy1, dx2, dy2 = det.xyxy
                dw, dh = max(1, dx2 - dx1), max(1, dy2 - dy1)
                d_area = float(dw * dh)

                # Prevent extreme scale jumps (e.g. tiny distant box hijacking a huge foreground track)
                area_ratio = min(t_area, d_area) / max(t_area, d_area)
                if area_ratio < 0.12:
                    continue

                iou_curr, ios_curr = overlap_scores(det, t_curr_det)
                iou_pred, ios_pred = overlap_scores(det, t_pred_det)
                best_iou = max(iou_curr, iou_pred)
                best_ios = max(ios_curr, ios_pred)

                dcx = (dx1 + dx2) / 2.0
                dcy = (dy1 + dy2) / 2.0
                dist_curr = math.hypot(tcx - dcx, tcy - dcy)
                dist_pred = math.hypot(pcx - dcx, pcy - dcy)
                best_dist = min(dist_curr, dist_pred)

                # Scale-aware center distance limit
                char_size = max(tw, th, dw, dh)
                allowed_dist = min(
                    self.max_center_distance,
                    max(45.0, 1.65 * char_size),
                )

                if (
                    best_iou >= self.iou_threshold
                    or (best_ios >= 0.45 and area_ratio >= 0.25)
                    or best_dist <= allowed_dist
                ):
                    dist_score = max(0.0, 1.0 - (best_dist / max(1.0, self.max_center_distance)))
                    score = (
                        2.5 * best_iou
                        + 0.8 * best_ios * area_ratio
                        + 0.7 * dist_score
                        + 0.3 * area_ratio
                    )
                    candidate_pairs.append((score, track_id, det_idx))

        # Greedy optimal one-to-one assignment
        candidate_pairs.sort(key=lambda item: item[0], reverse=True)
        matched_tracks: set[int] = set()
        matched_dets: dict[int, int] = {}  # det_idx -> track_id

        for _, track_id, det_idx in candidate_pairs:
            if track_id in matched_tracks or det_idx in matched_dets:
                continue
            matched_tracks.add(track_id)
            matched_dets[det_idx] = track_id

        assigned_detections: list[Detection] = []

        for det_idx, det in enumerate(detections):
            if det_idx in matched_dets:
                track_id = matched_dets[det_idx]
                track = self.active_tracks[track_id]
                dt = max(1, frame_index - track.last_frame)

                prev_cx = (track.xyxy[0] + track.xyxy[2]) / 2.0
                prev_cy = (track.xyxy[1] + track.xyxy[3]) / 2.0
                new_cx = (det.xyxy[0] + det.xyxy[2]) / 2.0
                new_cy = (det.xyxy[1] + det.xyxy[3]) / 2.0
                inst_vx = (new_cx - prev_cx) / dt
                inst_vy = (new_cy - prev_cy) / dt

                if track.hits == 1:
                    track.vx = inst_vx
                    track.vy = inst_vy
                else:
                    track.vx = 0.5 * track.vx + 0.5 * inst_vx
                    track.vy = 0.5 * track.vy + 0.5 * inst_vy

                track.xyxy = det.xyxy
                track.last_frame = frame_index
                track.hits += 1

                assigned_detections.append(
                    Detection(
                        class_id=det.class_id,
                        confidence=det.confidence,
                        xyxy=det.xyxy,
                        track_id=track_id,
                    )
                )
            else:
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
