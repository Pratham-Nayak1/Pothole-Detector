"""Unit tests for multi-scale Weighted Boxes Fusion (WBF) and tiling."""

from src.detection.postprocess import Detection
from src.detection.slicing import fuse_multiscale_detections, generate_tiles


def test_fuse_multiscale_detections_boosts_consensus_and_weights_coords() -> None:
    det_640 = Detection(class_id=3, confidence=0.30, xyxy=(100, 100, 200, 200))
    det_800 = Detection(class_id=3, confidence=0.20, xyxy=(104, 102, 204, 202))

    fused = fuse_multiscale_detections(
        [(det_640, 640), (det_800, 800)],
        fusion_iou=0.40,
        consensus_boost=0.22,
        num_scales=2,
    )

    assert len(fused) == 1
    assert fused[0].class_id == 3
    assert fused[0].confidence > 0.30
    assert 100 <= fused[0].xyxy[0] <= 104


def test_generate_tiles_covers_full_image_dimensions() -> None:
    tiles = generate_tiles(1280, 720, tile_size=640, overlap=0.25)

    assert tiles
    assert min(t[0] for t in tiles) == 0
    assert min(t[1] for t in tiles) == 0
    assert max(t[2] for t in tiles) == 1280
    assert max(t[3] for t in tiles) == 720
