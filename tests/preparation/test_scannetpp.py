"""
Prepare scene caches from a synthetic ScanNet++ scene, with and without an annotation package.
"""

import errno
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from egorecall.cli import prepare_scannetpp
from egorecall.data.scannetpp import ScanNetPPScene
from egorecall.data.scene_h5 import SceneH5
from egorecall.integrity import fingerprint_file
from egorecall.preparation.scannetpp import prepare_scene
from egorecall.validation.cache import validate_scene_cache


def test_preparation_without_annotations_needs_only_cache_inputs(
    raw_root: Path,
    tmp_path: Path,
    ffmpeg_path: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    Prepare through the CLI with no annotation root, mesh, or segmentation file.

    Args:
        raw_root: Synthetic source directory containing the cache's actual inputs.
        tmp_path: Configuration and cache parent.
        ffmpeg_path: FFmpeg executable.
        monkeypatch: Fixture supplying CLI arguments.
        capsys: Captured CLI output and argument errors.
    """
    # Remove assets that scene-cache preparation does not consume.
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

    # The annotation-based CLI still requires an annotation root.
    monkeypatch.setattr(sys, "argv", ["egorecall-prepare", "--config", str(config), "--split", "test"])
    with pytest.raises(SystemExit) as error:
        prepare_scannetpp.main()
    assert error.value.code == 2
    assert "requires dataset_root" in capsys.readouterr().err


def test_preparation_selects_scenes_by_stage(
    package_root: Path, raw_root: Path, tmp_path: Path, ffmpeg_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Prepare every scene of a stage selection through the CLI, using the frame names from the
    annotation package. Visibility files are not read during preparation.

    Args:
        package_root: Package whose stage 1 contains scene_a only; its visibility files will be removed.
        raw_root: Source download containing scene_a only.
        tmp_path: Configuration and cache parent.
        ffmpeg_path: FFmpeg executable.
        monkeypatch: Fixture supplying CLI arguments.
    """
    for path in (package_root / "annotations").iterdir():
        path.unlink()
    cache_root = tmp_path / "stage_cache"
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
            "--ffmpeg",
            ffmpeg_path,
        ],
    )
    prepare_scannetpp.main()

    # Stage 1 selects scene_a only, so scene_b is neither read from the source nor prepared.
    with SceneH5(cache_root / "scene_a.h5") as cache:
        assert cache.frame_names == ("frame_000000", "frame_000010", "frame_000020")
        assert cache.observation(1).depth[0, 0] == 1010
    assert not (cache_root / "scene_b.h5").exists()


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


def test_failed_preparation_leaves_no_completed_cache(raw_root: Path, tmp_path: Path, ffmpeg_path: str) -> None:
    """
    A missing depth frame aborts preparation and removes temporary outputs.

    Args:
        raw_root: Source scene whose depth timeline will be truncated cleanly.
        tmp_path: Cache parent.
        ffmpeg_path: FFmpeg executable.
    """
    source_scene = ScanNetPPScene(raw_root, "scene_a")
    depth = source_scene.iphone_depth_path.read_bytes()
    first_size = int.from_bytes(depth[:4], "little")
    source_scene.iphone_depth_path.write_bytes(depth[: 4 + first_size])

    root = tmp_path / "failed_cache"
    with pytest.raises(ValueError, match="depth is missing"):
        prepare_scene(source_scene, root, ffmpeg=ffmpeg_path)
    assert not list(root.iterdir())

    with pytest.raises(ValueError, match="outside the ScanNet"):
        prepare_scene(source_scene, raw_root / "cache", ffmpeg=ffmpeg_path)


@pytest.mark.parametrize("link_failure", ["unsupported", "published_by_another_run"])
def test_publishing_without_hard_links_or_after_another_run(
    raw_root: Path, tmp_path: Path, ffmpeg_path: str, monkeypatch: pytest.MonkeyPatch, link_failure: str
) -> None:
    """
    Publish with a rename where hard links are unsupported, and keep a cache that another run
    published first. Either way, one complete cache remains and no staging folder is left behind.

    Args:
        raw_root: Synthetic source scene.
        tmp_path: Cache parent.
        ffmpeg_path: FFmpeg executable.
        monkeypatch: Fixture replacing os.link.
        link_failure: How creating the hard link fails.
    """
    link = os.link

    def failing_link(source: Path, destination: Path) -> None:
        """
        Fail like a filesystem without hard links, or publish first and then fail like a concurrent run.

        Args:
            source: Completed temporary cache.
            destination: Path of the published cache.
        """
        if link_failure == "unsupported":
            raise OSError(errno.EPERM, "Operation not permitted")
        link(source, destination)
        raise FileExistsError(errno.EEXIST, "File exists")

    monkeypatch.setattr(os, "link", failing_link)
    cache_root = tmp_path / "published_cache"
    output = prepare_scene(ScanNetPPScene(raw_root, "scene_a"), cache_root, ffmpeg=ffmpeg_path)

    assert [path.name for path in cache_root.iterdir()] == [output.name] == ["scene_a.h5"]
    assert validate_scene_cache(output, decode_all=True) == 3


def test_scene_without_source_objects(raw_root: Path, tmp_path: Path, ffmpeg_path: str) -> None:
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


def test_preparation_uses_supplied_frame_names(raw_root: Path, tmp_path: Path, ffmpeg_path: str) -> None:
    """
    Supplied frame names select the same source positions for cameras, RGB, and depth.

    Args:
        raw_root: Source frames with timestamps and depth values encoding their position.
        tmp_path: Cache destination parent.
        ffmpeg_path: FFmpeg executable.
    """
    frame_names = ("frame_000001", "frame_000011")
    cache_path = prepare_scene(
        ScanNetPPScene(raw_root, "scene_a"),
        tmp_path / "selected_cache",
        frame_names=frame_names,
        ffmpeg=ffmpeg_path,
    )
    validate_scene_cache(cache_path, decode_all=True)
    with SceneH5(cache_path) as scene_h5:
        assert scene_h5.frame_names == frame_names
        for frame_idx, source_idx in enumerate((1, 11)):
            observation = scene_h5.observation(frame_idx)
            assert observation.depth[0, 0] == 1000 + source_idx
            np.testing.assert_allclose(observation.camera_to_world[0, 3], source_idx / 100)
            np.testing.assert_allclose(observation.rgb[2, 2], [200, 20 + source_idx, 30], atol=5)
