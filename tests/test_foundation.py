"""
Exercise raw-source preparation, cache integrity, and query-time observation boundaries.
"""

import json
import sys
import zlib
from dataclasses import asdict
from pathlib import Path

import h5py
import lz4.block
import numpy as np
import pytest

from egorecall import DatasetPaths
from egorecall.cli import prepare_scannetpp
from egorecall.data import EgoRecallAnnotations, EgoRecallDataset
from egorecall.data.check import check_dataset, verify_package
from egorecall.data.integrity import fingerprint_file
from egorecall.data.media import decode_image, extract_video_frames
from egorecall.data.prepare import prepare_scene
from egorecall.data.scannetpp import ScanNetPPScene, object_geometry_sha256, scale_intrinsics
from egorecall.data.scene_h5 import CACHE_VERSION, SceneH5
from scannetpp_common.iphone import iter_depth_frames


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
    annotations = EgoRecallAnnotations(package_root)
    root = tmp_path / "cache"
    prepare_scene(
        ScanNetPPScene(raw_root, "scene_a"),
        root,
        expected_frame_names=annotations.frame_names("scene_a"),
        ffmpeg=ffmpeg_path,
    )
    return root


def test_raw_geometry_and_camera_conventions(raw_root: Path) -> None:
    """
    Preserve source object membership, matrix orientation, timestamps, and intrinsic scaling.

    Args:
        raw_root: Scene with known source boxes and cameras.
    """
    source = ScanNetPPScene(raw_root, "scene_a")
    cameras = source.cameras()
    assert cameras.frame_names == ("frame_000000", "frame_000010", "frame_000020")
    np.testing.assert_array_equal(cameras.camera_to_world[:, 0, 3], [0, 0.1, 0.2])
    np.testing.assert_allclose(cameras.timestamps, 100 + np.array([0, 10, 20]) / 60)

    original = cameras.intrinsics.copy()
    scaled = scale_intrinsics(cameras.intrinsics, (32, 24), (256, 192))
    np.testing.assert_array_equal(scaled[0], [[160, 0, 128], [0, 160, 96], [0, 0, 1]])
    np.testing.assert_array_equal(cameras.intrinsics, original)

    objects = source.objects()
    assert set(objects) == {1, 2, 3}
    np.testing.assert_array_equal(objects[2].lengths, [1, 2, 3])
    assert objects[3].label == "lamp"

    assert source.metadata_path("semantic_classes.txt").is_file()
    assert source.paths.scan_mesh_path.is_file()
    assert source.paths.scan_mesh_segs_path.is_file()


@pytest.mark.parametrize("scene_id", ["../scene_a", "/scene_a", "missing"])
def test_raw_scene_paths_fail(raw_root: Path, scene_id: str) -> None:
    """
    Reject path traversal and missing source scenes.

    Args:
        raw_root: Original-layout source root.
        scene_id: Invalid or unavailable scene identifier.
    """
    with pytest.raises((ValueError, FileNotFoundError)):
        ScanNetPPScene(raw_root, scene_id)
    with pytest.raises(FileNotFoundError, match="must contain data/"):
        ScanNetPPScene(raw_root / "data", "scene_a")


def test_annotation_operations_require_dataset_root() -> None:
    """
    Report the missing annotation root before attempting package reads.
    """
    with pytest.raises(ValueError, match="dataset_root"):
        EgoRecallDataset(DatasetPaths())
    with pytest.raises(ValueError, match="dataset_root"):
        check_dataset(DatasetPaths())


def test_preparation_without_annotations_needs_only_cache_inputs(
    raw_root: Path,
    tmp_path: Path,
    ffmpeg_path: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    Prepare through the CLI with no annotation root, global metadata, mesh, or segmentation file.

    Args:
        raw_root: Synthetic source directory containing the cache's actual inputs.
        tmp_path: Configuration and cache parent.
        ffmpeg_path: FFmpeg executable.
        monkeypatch: Fixture supplying CLI arguments.
        capsys: Captured CLI output and argument errors.
    """
    # Remove assets that scene-cache preparation does not consume.
    metadata = raw_root / "metadata"
    saved_metadata = raw_root / "saved_metadata"
    metadata.rename(saved_metadata)
    for name in ("mesh_aligned_0.05.ply", "segments.json"):
        (raw_root / "data/scene_a/scans" / name).unlink()

    cache_root = tmp_path / "standalone_cache"
    config = tmp_path / "paths.toml"
    config.write_text(
        f"[paths]\nscannetpp_root = {json.dumps(str(raw_root))}\ncache_root = {json.dumps(str(cache_root))}\n"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "egorecall-prepare",
            "--config",
            str(config),
            "--without-annotations",
            "--scenes",
            "scene_a",
            "--subsample-factor",
            "10",
            "--ffmpeg",
            ffmpeg_path,
        ],
    )
    prepare_scannetpp.main()
    with SceneH5(cache_root / "scene_a.h5") as cache:
        assert cache.frame_names == ("frame_000000", "frame_000010", "frame_000020")
        assert cache.observation(1).depth[0, 0] == 1010
        assert set(cache.objects()) == {1, 2, 3}

    # Metadata is required only when a caller requests a particular metadata file.
    source = ScanNetPPScene(raw_root, "scene_a")
    with pytest.raises(FileNotFoundError, match=r"Missing ScanNet\+\+ metadata"):
        source.metadata_path("semantic_classes.txt")
    saved_metadata.rename(metadata)
    assert source.metadata_path("semantic_classes.txt") == metadata / "semantic_classes.txt"

    # The annotation-based CLI still requires an annotation root.
    monkeypatch.setattr(sys, "argv", ["egorecall-prepare", "--config", str(config), "--split", "test"])
    with pytest.raises(SystemExit) as error:
        prepare_scannetpp.main()
    assert error.value.code == 2
    assert "requires dataset_root" in capsys.readouterr().err


def test_preparation_uses_only_scene_metadata(
    package_root: Path, raw_root: Path, tmp_path: Path, ffmpeg_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Prepare an annotated scene without loading query contents or visibility files.

    Args:
        package_root: Metadata package whose query and visibility payloads will be removed.
        raw_root: Source scene matching the package's timeline.
        tmp_path: Configuration and cache parent.
        ffmpeg_path: FFmpeg executable.
        monkeypatch: Fixture supplying CLI arguments.
    """
    (package_root / "queries/test.parquet").unlink()
    for path in (package_root / "annotations").iterdir():
        path.unlink()
    cache_root = tmp_path / "metadata_cache"
    config = tmp_path / "paths.toml"
    config.write_text(
        f"[paths]\ndataset_root = {json.dumps(str(package_root))}\n"
        f"scannetpp_root = {json.dumps(str(raw_root))}\ncache_root = {json.dumps(str(cache_root))}\n"
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "egorecall-prepare",
            "--config",
            str(config),
            "--split",
            "test",
            "--stages",
            "1",
            "--scenes",
            "scene_a",
            "--ffmpeg",
            ffmpeg_path,
        ],
    )
    prepare_scannetpp.main()

    with SceneH5(cache_root / "scene_a.h5") as cache:
        assert cache.frame_names == ("frame_000000", "frame_000010", "frame_000020")
        assert cache.observation(1).depth[0, 0] == 1010
    assert not (cache_root / "scene_b.h5").exists()


def test_camera_access_is_independent_of_images(
    package_root: Path, prepared_cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Read matching camera metadata without image access and keep returned arrays independent.

    Args:
        package_root: Query selection for the cached scene.
        prepared_cache: Scene cache containing known camera values.
        monkeypatch: Fixture used to reject any image read or decode attempt.
    """
    from egorecall.data import scene_h5

    def unexpected_image_access(*args: object, **kwargs: object) -> None:
        """
        Fail if camera access tries to read or decode an image.

        Args:
            args: Positional image-access arguments.
            kwargs: Keyword image-access arguments.
        """
        raise AssertionError("Camera access must not read or decode images.")

    with SceneH5(prepared_cache / "scene_a.h5") as cache:
        expected = cache.observation(1)
        monkeypatch.setattr(SceneH5, "encoded_image", unexpected_image_access)
        monkeypatch.setattr(scene_h5, "decode_image", unexpected_image_access)

        camera = cache.camera(1)
        assert camera.frame_idx == expected.frame_idx
        assert camera.frame_name == expected.frame_name
        assert camera.timestamp == expected.timestamp
        for name in ("camera_to_world", "rgb_intrinsics", "depth_intrinsics"):
            np.testing.assert_array_equal(getattr(camera, name), getattr(expected, name))
            getattr(camera, name)[:] = 999
            np.testing.assert_array_equal(getattr(cache.camera(1), name), getattr(expected, name))

        for invalid in (-1, True, 1.0, 3):
            with pytest.raises((ValueError, IndexError)):
                cache.camera(invalid)

    with EgoRecallDataset(DatasetPaths(package_root, cache_root=prepared_cache)).open_scene("scene_a") as scene:
        window = scene.query(17).observations
        camera = window.camera(1)
        np.testing.assert_array_equal(camera.camera_to_world, expected.camera_to_world)
        assert camera.frame_name == "frame_000010"

        for invalid in (-1, True, 1.0, 2, 100):
            with pytest.raises((ValueError, IndexError)):
                window.camera(invalid)


def test_source_checks_need_only_cache_inputs(package_root: Path, raw_root: Path, prepared_cache: Path) -> None:
    """
    Check source/cache consistency without unrelated assets, while still requiring source annotations.

    Args:
        package_root: EgoRecall annotation package.
        raw_root: Source directory to reduce to the files consumed by preparation.
        prepared_cache: Compatible scene cache.
    """
    (raw_root / "metadata").rename(raw_root / "saved_metadata")
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


@pytest.mark.parametrize("codec", ["global", "lz4", "deflate", "mixed"])
def test_depth_formats_and_units(tmp_path: Path, codec: str) -> None:
    """
    Decode supported depth formats and retain exact integer values at boundaries.

    Args:
        tmp_path: Directory for encoded depth streams.
        codec: Upstream whole-stream or per-frame compression format.
    """
    frames = np.zeros((3, 192, 256), dtype=np.uint16)
    for index in range(3):
        frames[index, :, :128] = 1000 + index * 1000
        frames[index, :, 128:] = 5000 + index * 1000

    # Encode the same depth values using each supported source representation.
    floats = frames.astype("<f4") / 1000
    if codec == "global":
        encoded = zlib.compress(floats.tobytes(), wbits=-zlib.MAX_WBITS)
    else:
        blocks = []
        for index in range(3):
            if codec == "lz4" or (codec == "mixed" and index % 2 == 0):
                block = lz4.block.compress(frames[index].tobytes(), store_size=False)
            else:
                block = zlib.compress(floats[index].tobytes(), wbits=-zlib.MAX_WBITS)
            blocks.append(len(block).to_bytes(4, "little") + block)
        encoded = b"".join(blocks)

    path = tmp_path / "depth.bin"
    path.write_bytes(encoded)

    # Check the complete sequence and a selection that preserves source indices.
    decoded = list(iter_depth_frames(path))
    assert [index for index, _ in decoded] == [0, 1, 2]
    for index, depth in decoded:
        np.testing.assert_array_equal(depth, frames[index])

    selected = list(iter_depth_frames(path, selected={1, 2}))
    assert [index for index, _ in selected] == [1, 2]

    # Truncation must fail rather than returning a partial sequence as a successful decode.
    path.write_bytes(encoded[:-5])
    with pytest.raises(ValueError):
        list(iter_depth_frames(path))


def test_prepared_pixels_camera_and_reuse(raw_root: Path, prepared_cache: Path, ffmpeg_path: str) -> None:
    """
    Verify source frame alignment, RGB orientation/channels, depth boundaries, and cache reuse.

    Args:
        raw_root: Scene with known source pixel values.
        prepared_cache: Cache prepared from that scene.
        ffmpeg_path: FFmpeg executable for the preparation interface.
    """
    path = prepared_cache / "scene_a.h5"
    with SceneH5(path) as cache:
        frame = cache.observation(1)
        assert frame.frame_name == "frame_000010"

        assert frame.depth.dtype == np.uint16
        np.testing.assert_array_equal(frame.depth[:, :128], np.full((192, 128), 1010, dtype=np.uint16))
        np.testing.assert_array_equal(frame.depth[:, 128:], np.full((192, 128), 5010, dtype=np.uint16))

        np.testing.assert_allclose(frame.rgb[2, 2], [200, 30, 30], atol=5)
        np.testing.assert_allclose(frame.rgb[-3, -3], [30, 30, 200], atol=5)
        assert np.all(frame.mask[:, :16] == 255) and np.all(frame.mask[:, 16:] == 0)

        assert frame.camera_to_world[0, 3] == 0.1
        assert frame.depth_intrinsics[0, 0] == 160

        frame.camera_to_world[0, 3] = 999
        assert cache.observation(1).camera_to_world[0, 3] == 0.1

    before = (path.stat().st_mtime_ns, fingerprint_file(path))
    assert prepare_scene(ScanNetPPScene(raw_root, "scene_a"), prepared_cache, ffmpeg=ffmpeg_path) == path
    assert (path.stat().st_mtime_ns, fingerprint_file(path)) == before


def test_query_cutoff_and_separate_supervision(package_root: Path, raw_root: Path, prepared_cache: Path) -> None:
    """
    Expose only legal query history and keep full scene ground truth behind separate accessors.

    Args:
        package_root: Synthetic staged queries.
        raw_root: Source scene with three objects.
        prepared_cache: Full three-frame observation cache.
    """
    dataset = EgoRecallDataset(DatasetPaths(package_root, raw_root, prepared_cache))
    with dataset.open_scene("scene_a") as scene:
        sample = scene.query(17)
        assert set(asdict(sample.query)) == {"scene_id", "query_idx", "description", "frame"}
        assert sample.query.frame == 1

        assert sample.observations.frame_names == ("frame_000000", "frame_000010")
        assert [frame.frame_idx for frame in sample.observations] == [0, 1]
        assert sample.observations.frame(1).frame_name == "frame_000010"

        for invalid in (-1, True, 1.0, 2, 100):
            with pytest.raises((ValueError, IndexError)):
                sample.observations.frame(invalid)
            with pytest.raises((ValueError, IndexError)):
                sample.observations.encoded_image(invalid)

        assert scene.answer(17)["target_oids"] == [1]
        assert set(scene.supervision.source_objects) == {1, 2, 3}
        assert set(scene.supervision.filtered_objects) == {1, 2}
        assert scene.supervision.annotations["num_frames"] == 3

    with pytest.raises((ValueError, KeyError)):
        sample.observations.frame(0)


def test_stale_source_and_wrong_timeline_fail(raw_root: Path, prepared_cache: Path, ffmpeg_path: str) -> None:
    """
    Reject incompatible cache reuse without changing the existing completed file.

    Args:
        raw_root: Source scene to change after preparation.
        prepared_cache: Completed cache to preserve.
        ffmpeg_path: FFmpeg executable.
    """
    source = ScanNetPPScene(raw_root, "scene_a")
    before = fingerprint_file(prepared_cache / "scene_a.h5")
    with pytest.raises(ValueError, match="timeline"):
        prepare_scene(source, prepared_cache, subsample_factor=5, ffmpeg=ffmpeg_path)
    with pytest.raises(ValueError, match="frame mapping"):
        prepare_scene(source, prepared_cache, expected_frame_names=("frame_000000",), ffmpeg=ffmpeg_path)

    exif = source.paths.iphone_exif_path
    exif.write_text(exif.read_text() + "\n")
    with pytest.raises(ValueError, match="source files changed"):
        prepare_scene(source, prepared_cache, ffmpeg=ffmpeg_path)
    assert fingerprint_file(prepared_cache / "scene_a.h5") == before


def test_failed_preparation_leaves_no_completed_cache(raw_root: Path, tmp_path: Path, ffmpeg_path: str) -> None:
    """
    A missing depth frame aborts preparation and removes temporary outputs.

    Args:
        raw_root: Source scene whose depth timeline will be truncated cleanly.
        tmp_path: Cache parent.
        ffmpeg_path: FFmpeg executable.
    """
    source = ScanNetPPScene(raw_root, "scene_a")
    depth = source.paths.iphone_depth_path.read_bytes()
    first_size = int.from_bytes(depth[:4], "little")
    source.paths.iphone_depth_path.write_bytes(depth[: 4 + first_size])

    root = tmp_path / "failed_cache"
    with pytest.raises(ValueError, match="depth is missing"):
        prepare_scene(source, root, ffmpeg=ffmpeg_path)
    assert not list(root.iterdir())

    with pytest.raises(ValueError, match="outside the ScanNet"):
        prepare_scene(source, raw_root / "cache", ffmpeg=ffmpeg_path)


def test_corrupt_cached_payload_fails(prepared_cache: Path) -> None:
    """
    Detect changed encoded pixels before returning an observation.

    Args:
        prepared_cache: Completed cache whose second RGB payload will be replaced.
    """
    path = prepared_cache / "scene_a.h5"
    with h5py.File(path, "r+") as cache:
        cache["frames/rgb_jpg"][1] = cache["frames/rgb_jpg"][0]

    with SceneH5(path) as cache, pytest.raises(ValueError, match="checksum mismatch"):
        cache.observation(1)


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
        SceneH5(path)


@pytest.mark.parametrize("indices", [(7,), (1, 4, 20)])
def test_video_selection_uses_source_indices(
    raw_root: Path, tmp_path: Path, ffmpeg_path: str, indices: tuple[int, ...]
) -> None:
    """
    Select a nonzero singleton or irregular indices without shifting frame identities.

    Args:
        raw_root: Synthetic video whose green channel encodes source time.
        tmp_path: Extraction directory parent.
        ffmpeg_path: FFmpeg executable.
        indices: Exact source indices to extract.
    """
    video = raw_root / "data/scene_a/iphone/rgb.mkv"
    names = tuple(f"frame_{index:06d}" for index in indices)
    files = extract_video_frames(video, names, tmp_path / "selected", ffmpeg=ffmpeg_path)
    for index, path in zip(indices, files, strict=True):
        rgb = decode_image(path.read_bytes(), "rgb", (32, 24))
        np.testing.assert_allclose(rgb[2, 2], [200, 20 + index, 30], atol=5)

    with pytest.raises(ValueError, match="extracted 1 frames, expected 2"):
        extract_video_frames(video, ("frame_000000", "frame_000100"), tmp_path / "missing", ffmpeg=ffmpeg_path)


@pytest.mark.parametrize("change", ["missing", "label"])
def test_source_object_join_fails(
    package_root: Path, raw_root: Path, tmp_path: Path, ffmpeg_path: str, change: str
) -> None:
    """
    Reject cached objects that are missing from, or inconsistent with, the annotation population.

    Args:
        package_root: Two-object visibility annotation.
        raw_root: Source objects to corrupt.
        tmp_path: Cache parent.
        ffmpeg_path: FFmpeg executable.
        change: Remove a required object or change its label.
    """
    path = raw_root / "data/scene_a/scans/segments_anno.json"
    source = json.loads(path.read_text())
    if change == "missing":
        source["segGroups"] = source["segGroups"][1:]
    else:
        source["segGroups"][0]["label"] = "desk"
    path.write_text(json.dumps(source))

    cache_root = tmp_path / "changed_objects_cache"
    prepare_scene(ScanNetPPScene(raw_root, "scene_a"), cache_root, ffmpeg=ffmpeg_path)
    dataset = EgoRecallDataset(DatasetPaths(package_root, cache_root=cache_root))
    with dataset.open_scene("scene_a") as scene, pytest.raises((KeyError, ValueError)):
        scene.supervision

    _add_manifest_hashes(package_root)
    with pytest.raises((KeyError, ValueError)):
        check_dataset(DatasetPaths(package_root, cache_root=cache_root), scene_ids=["scene_a"], check_cache=True)


def test_observations_and_supervision_without_raw_source(package_root: Path, prepared_cache: Path) -> None:
    """
    Read observations, answers, and full scene supervision with no raw source configured.

    Args:
        package_root: Query annotations.
        prepared_cache: Existing observation cache.
    """
    dataset = EgoRecallDataset(DatasetPaths(package_root, cache_root=prepared_cache), stages=1)
    scene_id, query_idx = dataset.annotations.query_keys[0]
    with dataset.open_scene(scene_id) as scene:
        sample = scene.query(query_idx)
        assert len(sample.observations) == 1
        assert sample.observations.frame(0).depth[0, 0] == 1000
        with pytest.raises(AttributeError):
            sample.observations.frame_names = ("frame_000000", "frame_000010")
        assert scene.answer(query_idx)["target_oids"] == [1]
        assert set(scene.supervision.source_objects) == {1, 2, 3}
        assert set(scene.supervision.filtered_objects) == {1, 2}
        assert scene.supervision.source_objects[3].label == "lamp"
        with pytest.raises(KeyError):
            scene.query(17)
    with pytest.raises(KeyError):
        dataset.open_scene("scene_b")


def test_cached_geometry_survives_unavailable_raw_source(
    package_root: Path, raw_root: Path, prepared_cache: Path
) -> None:
    """
    Keep supervision and cache checks usable after the configured raw directory becomes unavailable.

    Args:
        package_root: Query annotations.
        raw_root: Source directory to move after preparation.
        prepared_cache: Completed scene cache.
    """
    source_objects = ScanNetPPScene(raw_root, "scene_a").objects()
    paths = DatasetPaths(package_root, raw_root, prepared_cache)
    raw_root.rename(raw_root.with_name("disconnected_source"))

    with EgoRecallDataset(paths).open_scene("scene_a") as scene:
        assert scene.query(17).observations.frame(1).frame_name == "frame_000010"
        assert set(scene.supervision.source_objects) == set(source_objects)
        for oid, expected in source_objects.items():
            actual = scene.supervision.source_objects[oid]
            assert actual.label == expected.label
            for name in ("centroid", "axes", "lengths", "minimum", "maximum"):
                np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))

    _add_manifest_hashes(package_root)
    report = check_dataset(paths, scene_ids=["scene_a"], check_cache=True, decode_all=True)
    assert report.source_scenes == 0 and report.cache_scenes == 1 and report.frames_decoded == 3


def test_cached_geometry_is_independently_owned(prepared_cache: Path) -> None:
    """
    Caller edits to object records must not change later cache lookups.

    Args:
        prepared_cache: Completed scene cache.
    """
    with SceneH5(prepared_cache / "scene_a.h5") as cache:
        objects = cache.objects()
        objects[1].centroid[:] = 999
        del objects[2]
        fresh = cache.objects()
        np.testing.assert_array_equal(fresh[1].centroid, [1, 0, 0])
        assert set(fresh) == {1, 2, 3}


def test_empty_source_object_population(raw_root: Path, tmp_path: Path, ffmpeg_path: str) -> None:
    """
    Preserve an explicitly empty source annotation instead of inventing geometry records.

    Args:
        raw_root: Scene whose source annotation will contain no objects.
        tmp_path: Cache parent.
        ffmpeg_path: FFmpeg executable.
    """
    path = raw_root / "data/scene_a/scans/segments_anno.json"
    path.write_text(json.dumps({"segGroups": []}))
    output = prepare_scene(ScanNetPPScene(raw_root, "scene_a"), tmp_path / "empty_objects", ffmpeg=ffmpeg_path)

    with SceneH5(output) as cache:
        assert cache.objects() == {}
        assert cache.observation(0).depth[0, 0] == 1000


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
    annotation = json.loads(path.read_text())
    annotation["segGroups"][0]["obb"]["centroid"][0] += 0.25
    path.write_text(json.dumps(annotation))
    before = fingerprint_file(prepared_cache / "scene_a.h5")

    with pytest.raises(ValueError, match="source files changed"):
        prepare_scene(ScanNetPPScene(raw_root, "scene_a"), prepared_cache, ffmpeg=ffmpeg_path)
    assert fingerprint_file(prepared_cache / "scene_a.h5") == before


@pytest.mark.parametrize("change", ["centroid", "label", "duplicate_id", "missing_geometry"])
def test_invalid_cached_objects_fail(prepared_cache: Path, change: str) -> None:
    """
    Reject missing, structurally invalid, or changed object data before it can be used as supervision.

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
        SceneH5(path)


def test_old_cache_schema_requires_recreation(prepared_cache: Path) -> None:
    """
    Fail explicitly on an observation-only cache instead of loading geometry from raw files.

    Args:
        prepared_cache: Cache to convert to the earlier incomplete schema.
    """
    path = prepared_cache / "scene_a.h5"
    with h5py.File(path, "r+") as cache:
        cache.attrs["schema_version"] = 1
        del cache["objects"]

    with pytest.raises(ValueError, match="recreate this cache"):
        SceneH5(path)


def _add_manifest_hashes(root: Path) -> None:
    """
    Fingerprint synthetic package files for checksum-checker tests.

    Args:
        root: Synthetic annotation package.
    """
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"] = {
        str(file.relative_to(root)): fingerprint_file(file)
        for file in root.rglob("*")
        if file.is_file() and file != path
    }
    path.write_text(json.dumps(manifest))


def test_checker_source_cache_and_package(package_root: Path, raw_root: Path, prepared_cache: Path) -> None:
    """
    Validate a complete package while limiting source/cache work to one available scene.

    Args:
        package_root: Two-scene annotation package.
        raw_root: Source download containing scene_a only.
        prepared_cache: Observation cache for scene_a.
    """
    _add_manifest_hashes(package_root)
    report = check_dataset(
        DatasetPaths(package_root, raw_root, prepared_cache),
        scene_ids=["scene_a"],
        check_source=True,
        check_cache=True,
        decode_all=True,
    )
    assert report.queries_checked == 4
    assert report.annotation_scenes == 2
    assert report.source_scenes == report.cache_scenes == 1
    assert report.frames_decoded == 3

    # Unlisted required payloads cannot bypass integrity verification.
    path = package_root / "manifest.json"
    manifest = json.loads(path.read_text())
    del manifest["files"]["queries/test.parquet"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="omits required files"):
        verify_package(package_root)

    # A listed payload must still match its recorded checksum after all required files are restored.
    _add_manifest_hashes(package_root)
    with (package_root / "scenes.json").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_package(package_root)


@pytest.mark.parametrize("name", ["objects", "any_target_queries"])
def test_checker_annotation_counts_fail(package_root: Path, name: str) -> None:
    """
    Compare manifest supervision totals with actual annotations and query values.

    Args:
        package_root: Synthetic annotation package.
        name: Manifest total to corrupt.
    """
    _add_manifest_hashes(package_root)
    path = package_root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["counts"][name] += 1
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=f"manifest/counts/{name}"):
        check_dataset(DatasetPaths(package_root))
