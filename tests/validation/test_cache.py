import json
from pathlib import Path

import h5py
import pytest

from egorecall import DatasetPaths
from egorecall.data.scannetpp import ScanNetPPScene
from egorecall.data.scene_h5 import CACHE_VERSION, SceneH5, object_geometry_sha256
from egorecall.integrity import fingerprint_file
from egorecall.validation.cache import validate_scene_cache
from egorecall.validation.check import check_dataset, check_source_scenes
from tests.helpers import _add_manifest_hashes


def test_stale_source_and_wrong_timeline_fail(raw_root: Path, prepared_cache: Path, ffmpeg_path: str) -> None:
    """
    The source/cache checker detects changed inputs; preparation leaves existing files untouched.

    Args:
        raw_root: Source scene to change after preparation.
        prepared_cache: Completed cache to preserve.
        ffmpeg_path: FFmpeg executable.
    """
    source_scene = ScanNetPPScene(raw_root, "scene_a")
    before = fingerprint_file(prepared_cache / "scene_a.h5")
    paths = DatasetPaths(scannetpp_root=raw_root, cache_root=prepared_cache)
    with pytest.raises(ValueError, match="timeline"):
        check_source_scenes(paths, ["scene_a"], 5, check_cache=True)

    exif = source_scene.iphone_exif_path
    exif.write_text(exif.read_text() + "\n")
    with pytest.raises(ValueError, match="source files changed"):
        check_source_scenes(paths, ["scene_a"], 10, check_cache=True)
    assert fingerprint_file(prepared_cache / "scene_a.h5") == before


def test_corrupt_cached_payload_fails(prepared_cache: Path) -> None:
    """
    Detect changed encoded pixels in the standalone checker, including an interior frame not selected for full decoding.

    Args:
        prepared_cache: Completed cache whose second RGB payload will be replaced.
    """
    path = prepared_cache / "scene_a.h5"
    with h5py.File(path, "r+") as cache:
        cache["frames/rgb_jpg"][1] = cache["frames/rgb_jpg"][0]

    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_scene_cache(path)


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("schema_version", float(CACHE_VERSION)),
        ("subsample_factor", 10.5),
        ("rgb_resolution", [32.5, 24.0]),
        ("scene_id", "../scene_a"),
        ("source_fps", "60"),
    ],
)
def test_invalid_cache_metadata_fails(prepared_cache: Path, attribute: str, value: object) -> None:
    """
    Reject malformed metadata instead of coercing it into apparently valid values.

    Args:
        prepared_cache: Cache whose metadata will be changed.
        attribute: Required attribute to corrupt.
        value: Invalid replacement value.
    """
    path = prepared_cache / "scene_a.h5"
    with h5py.File(path, "r+") as cache:
        cache.attrs[attribute] = value

    with pytest.raises(ValueError):
        validate_scene_cache(path)


def test_cache_geometry_is_compared_with_source(package_root: Path, raw_root: Path, prepared_cache: Path) -> None:
    """
    Explicit source checking detects different boxes even in a cache with a valid geometry checksum.

    Args:
        package_root: Annotation package with unchanged object IDs and labels.
        raw_root: Original source boxes.
        prepared_cache: Cache whose box centre will be changed consistently with its checksum.
    """
    path = prepared_cache / "scene_a.h5"
    with SceneH5(path) as cache:
        objects = cache.objects()
    objects[1].centroid[0] += 0.25
    with h5py.File(path, "r+") as cache:
        cache["objects/centroid"][0] = objects[1].centroid
        cache["objects"].attrs["sha256"] = object_geometry_sha256(objects)

    _add_manifest_hashes(package_root)
    paths = DatasetPaths(package_root, raw_root, prepared_cache)
    assert check_dataset(paths, scene_ids=["scene_a"], check_cache=True).cache_scenes == 1
    with pytest.raises(ValueError, match="cached object geometry differs"):
        check_dataset(paths, scene_ids=["scene_a"], check_cache=True, check_source=True)


def test_source_geometry_changes_invalidate_cache(raw_root: Path, prepared_cache: Path, ffmpeg_path: str) -> None:
    """
    Reject reuse when source box values change even though the observation files are unchanged.

    Args:
        raw_root: Source scene to edit after preparation.
        prepared_cache: Completed cache to preserve.
        ffmpeg_path: FFmpeg executable.
    """
    path = raw_root / "data/scene_a/scans/segments_anno.json"
    scene_annotations = json.loads(path.read_text())
    scene_annotations["segGroups"][0]["obb"]["centroid"][0] += 0.25
    path.write_text(json.dumps(scene_annotations))
    before = fingerprint_file(prepared_cache / "scene_a.h5")

    with pytest.raises(ValueError, match="source files changed"):
        check_source_scenes(
            DatasetPaths(scannetpp_root=raw_root, cache_root=prepared_cache), ["scene_a"], 10, check_cache=True
        )
    assert fingerprint_file(prepared_cache / "scene_a.h5") == before


@pytest.mark.parametrize("change", ["centroid", "label", "duplicate_id", "missing_geometry"])
def test_invalid_cached_objects_fail(prepared_cache: Path, change: str) -> None:
    """
    Reject missing, structurally invalid, or changed object data during the explicit cache check.

    Args:
        prepared_cache: Cache to corrupt.
        change: Object-table change to introduce.
    """
    path = prepared_cache / "scene_a.h5"
    with h5py.File(path, "r+") as cache:
        if change == "centroid":
            cache["objects/centroid"][0, 0] += 0.25
        elif change == "label":
            cache["objects/label"][0] = "desk"
        elif change == "duplicate_id":
            cache["objects/object_id"][1] = 1
        else:
            del cache["objects"]

    with pytest.raises((ValueError, KeyError)):
        validate_scene_cache(path)


def test_schema_1_cache_requires_recreation(prepared_cache: Path) -> None:
    """
    Fail explicitly on a schema-1 cache without object geometry instead of loading geometry from raw files.

    Args:
        prepared_cache: Cache rewritten as schema 1 with its objects group removed.
    """
    path = prepared_cache / "scene_a.h5"
    with h5py.File(path, "r+") as cache:
        cache.attrs["schema_version"] = 1
        del cache["objects"]

    with pytest.raises(ValueError, match="recreate this cache"):
        validate_scene_cache(path)
