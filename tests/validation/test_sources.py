"""
Check raw ScanNet++ scene paths and files, and compare source or cached objects with the visibility annotations.
"""

import json
from pathlib import Path

import pytest

from egorecall import DatasetPaths
from egorecall.data.scannetpp import ScanNetPPScene
from egorecall.preparation.scannetpp import prepare_scene
from egorecall.validation.check import check_dataset
from egorecall.validation.sources import validate_source_scene
from tests.helpers import _add_manifest_hashes


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
    _add_manifest_hashes(package_root)
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
def test_source_object_join_fails(
    package_root: Path, raw_root: Path, tmp_path: Path, ffmpeg_path: str, change: str, message: str
) -> None:
    """
    Reject cached objects that are missing from, or inconsistent with, the annotation population.

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
    _add_manifest_hashes(package_root)
    with pytest.raises(ValueError, match=message):
        check_dataset(DatasetPaths(package_root, cache_root=cache_root), scene_ids=["scene_a"], check_cache=True)
