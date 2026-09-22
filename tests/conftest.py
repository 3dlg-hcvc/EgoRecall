"""
Synthetic datasets with noncontiguous query IDs, shared IDs across scenes,
and several stages for testing data access and selection.
"""

import gzip
import json
import shutil
import subprocess
from pathlib import Path

import lz4.block
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from egorecall.data.scannetpp import ScanNetPPScene
from egorecall.data.schema import FRAME_SCHEMA, QUERY_SCHEMA, STAGE_SCHEMA
from egorecall.preparation.scannetpp import prepare_scene


@pytest.fixture
def package_root(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    """
    Build a four-query package with deliberately sparse scene-scoped IDs.
    An indirect parameter can select unstaged train instead of staged test.

    Args:
        tmp_path: Isolated directory provided by pytest.
        request: Fixture request with an optional split parameter.

    Returns:
        The synthetic package root.
    """
    split = getattr(request, "param", "test")
    root = tmp_path / "package"
    for directory in ("queries", "stages", "frames", "annotations"):
        (root / directory).mkdir(parents=True)

    # Query IDs are not row positions; ID 4 belongs to two different scenes.
    queries = []
    for scene_id, query_idx, frame in (("scene_a", 4, 0), ("scene_a", 17, 1), ("scene_a", 103, 2), ("scene_b", 4, 1)):
        queries.append(
            {
                "scene_id": scene_id,
                "query_idx": query_idx,
                "split": split,
                "description": "the first chair",
                "program_json": json.dumps(["first_seen", "chair"]),
                "source_query_id": f"source_{query_idx}",
                "program_depth": 1,
                "frame": frame,
                "any_target": False,
                "emit_reason": "new",
                "target_oids": [1],
                "visible_target_oids": [1] if frame < 2 else [],
                "hidden_target_oids": [1] if frame == 2 else [],
            }
        )
    pq.write_table(pa.Table.from_pylist(queries, schema=QUERY_SCHEMA), root / "queries" / f"{split}.parquet")

    # Assign stages independently of query order and preserve a second scene's key.
    assignments = [
        {"scene_id": scene_id, "query_idx": query_idx, "split": split, "stage": stage}
        for scene_id, query_idx, stage in (
            ("scene_b", 4, 2),
            ("scene_a", 103, 3),
            ("scene_a", 4, 1),
            ("scene_a", 17, 2),
        )
    ]
    if split != "train":
        pq.write_table(pa.Table.from_pylist(assignments, schema=STAGE_SCHEMA), root / "stages" / f"{split}.parquet")

    # Store complete frame timelines for both scenes.
    frames = [
        {"scene_id": scene_id, "frame_idx": index, "frame_name": f"frame_{index * 10:06d}"}
        for scene_id in ("scene_a", "scene_b")
        for index in range(3)
    ]
    pq.write_table(pa.Table.from_pylist(frames, schema=FRAME_SCHEMA), root / "frames" / f"{split}.parquet")

    # Retain a contextual object that never appears in these query answers.
    scenes = []
    for scene_id, num_queries in (("scene_a", 3), ("scene_b", 1)):
        scenes.append(
            {
                "scene_id": scene_id,
                "split": split,
                "num_frames": 3,
                "num_objects": 2,
                "num_queries": num_queries,
                "source_fps": 60.0,
                "subsample_factor": 10,
                "nominal_timeline_fps": 6.0,
                "annotations": f"annotations/{scene_id}.json.gz",
            }
        )

        objects = {
            str(oid): {
                "label": label,
                "visibility_segments": [[0, 1]],
                "per_frame": {str(frame): {"visible_area_frac": 0.5, "visible_pixels_frac": 0.1} for frame in (0, 1)},
                "temporal": {
                    "first_seen_frame": 0,
                    "last_seen_frame": 1,
                    "peak_visibility_frame": 0,
                    "total_visible_frames": 2,
                    "peak_visible_area_frac": 0.5,
                },
            }
            for oid, label in ((1, "chair"), (2, "table"))
        }
        annotation = {
            "schema_version": 1,
            "scene_id": scene_id,
            "num_frames": 3,
            "image_pixels": 100,
            "visibility_filter": "visibility_filter_v1",
            "objects": objects,
        }
        (root / "annotations" / f"{scene_id}.json.gz").write_bytes(gzip.compress(json.dumps(annotation).encode()))

    # Summarize the scene inventory and table counts after writing all payloads.
    (root / "scenes.json").write_text(json.dumps(scenes))
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "selection": {"split": split, "stage_from": 1, "stage_to": 3},
                "counts": {
                    "queries": 4,
                    "stage_assignments": 0 if split == "train" else 4,
                    "frames": 6,
                    "scenes": 2,
                    "objects": 4,
                    "any_target_queries": 0,
                },
            }
        )
    )
    return root


@pytest.fixture
def ffmpeg_path() -> str:
    """
    Require FFmpeg for synthetic source-video preparation.

    Returns:
        Executable path from the active environment.
    """
    executable = shutil.which("ffmpeg")
    if executable is None:
        pytest.fail("FFmpeg is required; install the foundation environment before running these tests.")
    return executable


@pytest.fixture
def raw_root(tmp_path: Path, ffmpeg_path: str) -> Path:
    """
    Build a small original-layout ScanNet++ scene with distinguishable RGB-D frames.

    Args:
        tmp_path: Isolated directory for synthetic source data.
        ffmpeg_path: FFmpeg executable used to encode lossless test videos.

    Returns:
        Dataset root containing data/scene_a and metadata.
    """
    root = tmp_path / "scannetpp"
    scene = root / "data/scene_a"
    iphone = scene / "iphone"
    scans = scene / "scans"
    for directory in (iphone, scans, root / "metadata"):
        directory.mkdir(parents=True)
    (root / "metadata/semantic_classes.txt").write_text("chair\ntable\nlamp\n")

    # Source indices 0, 10, and 20 become the package's three canonical frames.
    poses = {}
    rgb = np.zeros((21, 24, 32, 3), dtype=np.uint8)
    masks = np.zeros((21, 24, 32), dtype=np.uint8)
    depth_blocks = []
    for index in range(21):
        pose = np.eye(4)
        pose[0, 3] = index / 100
        poses[f"frame_{index:06d}"] = {
            "aligned_pose": pose.tolist(),
            "intrinsic": [[20, 0, 16], [0, 20, 12], [0, 0, 1]],
            "timestamp": 100.0 + index / 60,
        }

        rgb[index, :12] = [200, 20 + index, 30]
        rgb[index, 12:] = [30, 20 + index, 200]
        masks[index, :, :16] = 255

        depth = np.full((192, 256), 1000 + index, dtype="<u2")
        depth[:, 128:] = 5000 + index
        block = lz4.block.compress(depth.tobytes(), store_size=False)
        depth_blocks.append(len(block).to_bytes(4, "little") + block)

    (iphone / "pose_intrinsic_imu.json").write_text(json.dumps(poses))
    (iphone / "exif.json").write_text(json.dumps({"100.0": {"PixelXDimension": 32, "PixelYDimension": 24}}))
    (iphone / "depth.bin").write_bytes(b"".join(depth_blocks))

    # Lossless source videos make frame-index, channel-order, and orientation checks independent of the cache.
    for name, frames, pixel_format in (("rgb", rgb, "rgb24"), ("rgb_mask", masks, "gray")):
        subprocess.run(
            [
                ffmpeg_path,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pixel_format",
                pixel_format,
                "-video_size",
                "32x24",
                "-framerate",
                "60",
                "-i",
                "pipe:0",
                "-c:v",
                "ffv1",
                "-threads",
                "1",
                str(iphone / f"{name}.mkv"),
            ],
            input=frames.tobytes(),
            check=True,
            capture_output=True,
        )

    # Source geometry includes one object omitted from the visibility annotations.
    objects = []
    for oid, label in ((1, "chair"), (2, "table"), (3, "lamp")):
        objects.append(
            {
                "objectId": oid,
                "label": label,
                "segments": [oid - 1],
                "obb": {
                    "centroid": [oid, 0, 0],
                    "axesLengths": [1, 2, 3],
                    "normalizedAxes": np.eye(3).ravel().tolist(),
                    "min": [oid - 0.5, -1, -1.5],
                    "max": [oid + 0.5, 1, 1.5],
                },
            }
        )
    (scans / "segments_anno.json").write_text(json.dumps({"segGroups": objects}))
    (scans / "segments.json").write_text(json.dumps({"segIndices": [0, 1, 2]}))
    (scans / "mesh_aligned_0.05.ply").write_text("ply\nformat ascii 1.0\nelement vertex 0\nend_header\n")
    return root


@pytest.fixture
def prepared_cache(raw_root: Path, package_root: Path, tmp_path: Path, ffmpeg_path: str) -> Path:
    """
    Prepare the synthetic scene using the package's actual frame mapping.

    Args:
        raw_root: Synthetic original-layout source scene.
        package_root: Matching three-frame annotation package.
        tmp_path: Isolated writable cache parent.
        ffmpeg_path: FFmpeg executable.

    Returns:
        Cache directory containing scene_a.h5.
    """
    root = tmp_path / "cache"
    prepare_scene(
        ScanNetPPScene(raw_root, "scene_a"),
        root,
        ffmpeg=ffmpeg_path,
    )
    return root
