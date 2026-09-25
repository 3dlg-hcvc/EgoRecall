"""
Check query-time observation windows, separate ground-truth access, and scene selection
through EgoRecallDataset and EgoRecallScene.
"""

import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from egorecall import DatasetPaths
from egorecall.data import EgoRecallDataset
from egorecall.data.scannetpp import ScanNetPPScene
from egorecall.validation.check import check_dataset
from tests.helpers import _add_manifest_hashes


def test_annotation_operations_require_dataset_root() -> None:
    """
    Report the missing annotation root before attempting package reads.
    """
    with pytest.raises(ValueError, match="dataset_root"):
        EgoRecallDataset(DatasetPaths())
    with pytest.raises(ValueError, match="dataset_root"):
        check_dataset(DatasetPaths())


def test_query_cutoff_and_separate_supervision(package_root: Path, raw_root: Path, prepared_cache: Path) -> None:
    """
    Expose only legal query history and keep full scene ground truth behind separate accessors.

    Args:
        package_root: Synthetic staged queries.
        raw_root: Source scene with three objects.
        prepared_cache: Full three-frame observation cache.
    """
    dataset = EgoRecallDataset(DatasetPaths(package_root, raw_root, prepared_cache))
    with dataset.open_scene("scene_a") as scene_data:
        sample = scene_data.query(17)
        assert set(asdict(sample.query)) == {"scene_id", "query_idx", "description", "frame"}
        assert sample.query.frame == 1

        assert sample.observations.frame_names == ("frame_000000", "frame_000010")
        assert [frame.frame_idx for frame in sample.observations] == [0, 1]
        assert sample.observations.frame(1).frame_name == "frame_000010"

        for invalid in (-3, 1.0, 2, 100):
            with pytest.raises((TypeError, IndexError)):
                sample.observations.frame(invalid)
            with pytest.raises((TypeError, IndexError)):
                sample.observations.encoded_image(invalid)

        assert scene_data.answer(17)["target_oids"] == [1]
        assert set(scene_data.supervision.source_objects) == {1, 2, 3}
        assert set(scene_data.supervision.filtered_objects) == {1, 2}
        assert scene_data.supervision.annotations["num_frames"] == 3

    with pytest.raises((ValueError, KeyError)):
        sample.observations.frame(0)


def test_observations_and_supervision_without_raw_source(package_root: Path, prepared_cache: Path) -> None:
    """
    Read observations, answers, and full scene supervision with no raw source configured.

    Args:
        package_root: Query annotations.
        prepared_cache: Existing observation cache.
    """
    dataset = EgoRecallDataset(DatasetPaths(package_root, cache_root=prepared_cache), stages=1)
    scene_id, query_idx = dataset.annotations.query_keys[0]
    with dataset.open_scene(scene_id) as scene_data:
        sample = scene_data.query(query_idx)
        assert len(sample.observations) == 1
        assert sample.observations.frame(0).depth[0, 0] == 1000
        with pytest.raises(AttributeError):
            sample.observations.frame_names = ("frame_000000", "frame_000010")
        assert scene_data.answer(query_idx)["target_oids"] == [1]
        assert set(scene_data.supervision.source_objects) == {1, 2, 3}
        assert set(scene_data.supervision.filtered_objects) == {1, 2}
        assert scene_data.supervision.source_objects[3].label == "lamp"
        with pytest.raises(KeyError):
            scene_data.query(17)

    # Scene B has no stage-1 query; with every stage selected, it is valid but has no prepared cache.
    with pytest.raises(KeyError, match="scene_b"):
        dataset.open_scene("scene_b")
    with pytest.raises(FileNotFoundError):
        EgoRecallDataset(DatasetPaths(package_root, cache_root=prepared_cache)).open_scene("scene_b")


def test_scenes_outside_the_selection_are_rejected(package_root: Path, prepared_cache: Path) -> None:
    """
    Reject a scene without selected queries before opening its cache, even when a cache file exists.

    Args:
        package_root: Package whose stage 1 contains queries from scene_a only.
        prepared_cache: Cache directory holding scene_a.h5.
    """
    shutil.copy(prepared_cache / "scene_a.h5", prepared_cache / "scene_b.h5")
    dataset = EgoRecallDataset(DatasetPaths(package_root, cache_root=prepared_cache), stages=1)
    for scene_id in ("scene_b", "unknown_scene"):
        with pytest.raises(KeyError, match="no queries in the selected split and stages"):
            dataset.open_scene(scene_id)


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

    with EgoRecallDataset(paths).open_scene("scene_a") as scene_data:
        assert scene_data.query(17).observations.frame(1).frame_name == "frame_000010"
        assert set(scene_data.supervision.source_objects) == set(source_objects)
        for oid, expected in source_objects.items():
            actual = scene_data.supervision.source_objects[oid]
            assert actual.label == expected.label
            for name in ("centroid", "axes", "lengths", "minimum", "maximum"):
                np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))

    _add_manifest_hashes(package_root)
    report = check_dataset(paths, scene_ids=["scene_a"], check_cache=True, decode_all=True)
    assert report.source_scenes == 0 and report.cache_scenes == 1 and report.frames_decoded == 3


def test_negative_window_indices_stay_before_query(package_root: Path, prepared_cache: Path) -> None:
    """
    Negative indices count from the query frame, never from the end of the full cache.

    Args:
        package_root: Queries with a cutoff before the last cached frame.
        prepared_cache: Three-frame scene cache.
    """
    dataset = EgoRecallDataset(DatasetPaths(package_root, cache_root=prepared_cache))
    with dataset.open_scene("scene_a") as scene_data:
        observation_window = scene_data.query(17).observations
        assert observation_window.frame(-1).frame_idx == 1
        assert observation_window.camera(-1).frame_idx == 1
        assert observation_window.encoded_image(-1) == observation_window.encoded_image(1)
        assert observation_window.frame(-2).frame_idx == 0
        for accessor in (observation_window.frame, observation_window.camera, observation_window.encoded_image):
            with pytest.raises(IndexError):
                accessor(-3)
            with pytest.raises(IndexError):
                accessor(2)
