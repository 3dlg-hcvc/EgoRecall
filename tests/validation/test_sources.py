"""
Check raw ScanNet++ scene paths and files, and compare source or cached objects with the visibility annotations.
"""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from egorecall import DatasetPaths
from egorecall.data.scannetpp import ScanNetPPScene
from egorecall.geometry import CameraSequence, ObjectGeometry
from egorecall.preparation.scannetpp import prepare_scene
from egorecall.validation.check import check_dataset
from egorecall.validation.sources import validate_cameras, validate_object_geometry, validate_source_scene
from tests.helpers import add_manifest_hashes


@pytest.mark.parametrize("scene_id", ["../scene_a", "/scene_a", "missing"])
def test_raw_scene_paths_fail(raw_root: Path, scene_id: str) -> None:
    """
    Reject path traversal and missing source scenes.

    Args:
        raw_root: Original-layout source root.
        scene_id: Invalid or unavailable scene identifier.
    """
    with pytest.raises((ValueError, FileNotFoundError)):
        validate_source_scene(ScanNetPPScene(raw_root, scene_id), 10)
    with pytest.raises(FileNotFoundError):
        validate_source_scene(ScanNetPPScene(raw_root / "data", "scene_a"), 10)


def test_source_checks_need_only_cache_inputs(package_root: Path, raw_root: Path, prepared_cache: Path) -> None:
    """
    Check source/cache consistency without unrelated assets, while still requiring source annotations.

    Args:
        package_root: EgoRecall annotation package.
        raw_root: Source directory to reduce to the files consumed by preparation.
        prepared_cache: Compatible scene cache.
    """
    for name in ("mesh_aligned_0.05.ply", "segments.json"):
        (raw_root / "data/scene_a/scans" / name).unlink()
    add_manifest_hashes(package_root)
    paths = DatasetPaths(package_root, raw_root, prepared_cache)

    report = check_dataset(paths, scene_ids=["scene_a"], check_source=True, check_cache=True, decode_all=True)
    assert report.source_scenes == report.cache_scenes == 1
    assert report.frames_decoded == 3

    (raw_root / "data/scene_a/scans/segments_anno.json").unlink()
    with pytest.raises(FileNotFoundError, match="segments_anno.json"):
        check_dataset(paths, scene_ids=["scene_a"], check_source=True, check_cache=True)


@pytest.mark.parametrize(
    ("change", "message"), [("missing", "annotated objects have no ScanNet"), ("label", "labels differ")]
)
def test_annotated_object_mismatch_fails(
    package_root: Path, raw_root: Path, tmp_path: Path, ffmpeg_path: str, change: str, message: str
) -> None:
    """
    Reject a cache that lacks an annotated object or gives it a different label.

    Args:
        package_root: Two-object visibility annotation.
        raw_root: Source objects to corrupt.
        tmp_path: Cache parent.
        ffmpeg_path: FFmpeg executable.
        change: Remove a required object or change its label.
        message: Expected error text.
    """
    path = raw_root / "data/scene_a/scans/segments_anno.json"
    source_annotation = json.loads(path.read_text())
    if change == "missing":
        source_annotation["segGroups"] = source_annotation["segGroups"][1:]
    else:
        source_annotation["segGroups"][0]["label"] = "desk"
    path.write_text(json.dumps(source_annotation))

    cache_root = tmp_path / "changed_objects_cache"
    prepare_scene(ScanNetPPScene(raw_root, "scene_a"), cache_root, ffmpeg=ffmpeg_path)
    add_manifest_hashes(package_root)
    with pytest.raises(ValueError, match=message):
        check_dataset(DatasetPaths(package_root, cache_root=cache_root), scene_ids=["scene_a"], check_cache=True)


def test_source_frames_must_match_the_annotation_frames(package_root: Path, raw_root: Path) -> None:
    """
    Check source files without a cache, then reject a sampling stride whose source frames differ
    from the package's frame table.

    Args:
        package_root: Package whose scene_a sampling stride will change from 10 to 5.
        raw_root: Synthetic source scene with 21 pose records.
    """
    add_manifest_hashes(package_root)
    paths = DatasetPaths(package_root, raw_root)
    report = check_dataset(paths, scene_ids=["scene_a"], check_source=True)
    assert (report.source_scenes, report.cache_scenes, report.frames_decoded) == (1, 0, 0)

    path = package_root / "scenes.json"
    scenes = json.loads(path.read_text())
    scenes[0]["subsample_factor"] = 5
    path.write_text(json.dumps(scenes))
    add_manifest_hashes(package_root)
    with pytest.raises(ValueError, match="source frame names differ from the annotation frame mapping"):
        check_dataset(paths, scene_ids=["scene_a"], check_source=True)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("seg_groups", "segGroups must be a list"),
        ("duplicate_object", "Duplicate source objectId: 1"),
        ("exif", "one consistent RGB resolution"),
    ],
)
def test_invalid_source_records_fail(raw_root: Path, change: str, message: str) -> None:
    """
    Reject a source annotation whose objects are not a list or repeat an objectId, and EXIF records
    that disagree on the RGB resolution.

    Args:
        raw_root: Synthetic source scene to change.
        change: Invalid source record to introduce.
        message: Expected error text.
    """
    source_scene = ScanNetPPScene(raw_root, "scene_a")
    if change == "exif":
        exif = json.loads(source_scene.iphone_exif_path.read_text())
        exif["200.0"] = {"PixelXDimension": 64, "PixelYDimension": 48}
        source_scene.iphone_exif_path.write_text(json.dumps(exif))
    else:
        source_annotation = json.loads(source_scene.scan_anno_json_path.read_text())
        if change == "seg_groups":
            source_annotation["segGroups"] = {}
        else:
            source_annotation["segGroups"][1]["objectId"] = 1
        source_scene.scan_anno_json_path.write_text(json.dumps(source_annotation))

    with pytest.raises(ValueError, match=message):
        validate_source_scene(source_scene, 10)


def valid_cameras() -> CameraSequence:
    """
    Build two structurally valid camera records.

    Returns:
        Identity poses, pinhole intrinsics with 20-pixel focal lengths, and increasing timestamps.
    """
    return CameraSequence(
        frame_names=("frame_000000", "frame_000010"),
        camera_to_world=np.stack([np.eye(4)] * 2),
        intrinsics=np.stack([np.array([[20.0, 0, 16], [0, 20, 12], [0, 0, 1]])] * 2),
        timestamps=np.array([100.0, 100.5]),
        image_size=(32, 24),
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("order", "ordered by source index"),
        ("shape", "finite with shape"),
        ("finite", "finite with shape"),
        ("homogeneous", "homogeneous camera-to-world"),
        ("orthonormal", "rotations must be orthonormal"),
        ("handedness", "preserve handedness"),
        ("pinhole", "pinhole matrices"),
        ("focal", "Focal lengths must be positive"),
        ("timestamps", "strictly increasing"),
        ("image_size", "RGB dimension"),
    ],
)
def test_invalid_cameras_fail(change: str, message: str) -> None:
    """
    Reject camera records that are out of order, misshapen, not finite, not rigid right-handed poses,
    not pinhole matrices with positive focal lengths, not increasing in time, or without a valid image size.

    Args:
        change: Invalid camera value to introduce.
        message: Expected error text.
    """
    cameras = valid_cameras()
    validate_cameras(cameras)

    if change == "order":
        cameras = replace(cameras, frame_names=cameras.frame_names[::-1])
    elif change == "shape":
        cameras = replace(cameras, timestamps=np.array([100.0, 100.5, 101.0]))
    elif change == "finite":
        cameras.camera_to_world[0, 0, 3] = np.nan
    elif change == "homogeneous":
        cameras.camera_to_world[0, 3, 0] = 1.0
    elif change == "orthonormal":
        cameras.camera_to_world[0, 0, 0] = 2.0
    elif change == "handedness":
        cameras.camera_to_world[0, 2, 2] = -1.0
    elif change == "pinhole":
        cameras.intrinsics[0, 2, 0] = 1.0
    elif change == "focal":
        cameras.intrinsics[0, 0, 0] = -20.0
    elif change == "timestamps":
        cameras = replace(cameras, timestamps=np.array([100.0, 100.0]))
    else:
        cameras = replace(cameras, image_size=(0, 24))

    with pytest.raises(ValueError, match=message):
        validate_cameras(cameras)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("vector_shape", "invalid bounding-box vectors"),
        ("nonfinite", "invalid bounding-box vectors"),
        ("negative_length", "invalid bounding-box extents"),
        ("inverted_extent", "invalid bounding-box extents"),
        ("axes", "orthonormal rows"),
    ],
)
def test_invalid_boxes_fail(change: str, message: str) -> None:
    """
    Reject boxes with misshapen or non-finite vectors, negative lengths, a minimum above the
    maximum, or axes that are not orthonormal.

    Args:
        change: Invalid box value to introduce.
        message: Expected error text.
    """
    box = ObjectGeometry(1, "chair", np.zeros(3), np.eye(3), np.ones(3), np.full(3, -0.5), np.full(3, 0.5))
    validate_object_geometry(box, "scene_a/1")

    if change == "vector_shape":
        box = replace(box, centroid=np.zeros(2))
    elif change == "nonfinite":
        box.lengths[0] = np.inf
    elif change == "negative_length":
        box.lengths[0] = -1.0
    elif change == "inverted_extent":
        box.minimum[0] = 1.0
    else:
        box = replace(box, axes=2 * np.eye(3))

    with pytest.raises(ValueError, match=message):
        validate_object_geometry(box, "scene_a/1")
