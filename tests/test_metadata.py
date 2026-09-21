"""
Check preparation scene selection using metadata, stage assignments, and selected frame mappings.
"""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from egorecall.data import EgoRecallAnnotations
from egorecall.data.metadata import load_scene_metadata


def test_metadata_selection_matches_annotation_reader(package_root: Path) -> None:
    """
    Preserve exact-stage/range semantics and complete frame mappings across both reading paths.

    Args:
        package_root: Package with three stages, two scenes, and reordered assignments.
    """
    for stages in (None, 1, 2, 3, "1:3", "2:3"):
        annotations = EgoRecallAnnotations(package_root, stages=stages)
        scenes = load_scene_metadata(package_root, "test", stages=stages)
        assert tuple(scenes) == annotations.scene_ids
        for scene_id, scene in scenes.items():
            assert scene.metadata == annotations.get_scene(scene_id)
            assert scene.frame_names == annotations.frame_names(scene_id)

    selected = load_scene_metadata(package_root, "test", stages=2, scene_ids=["scene_b", "scene_a"])
    assert tuple(selected) == ("scene_b", "scene_a")


def test_query_and_visibility_files_are_not_needed(package_root: Path) -> None:
    """
    Load preparation metadata without reading query contents or visibility payloads.

    Args:
        package_root: Package from which query/visibility files will be removed.
    """
    (package_root / "queries/test.parquet").unlink()
    for path in (package_root / "annotations").iterdir():
        path.unlink()

    scenes = load_scene_metadata(package_root, "test", stages=1)
    assert tuple(scenes) == ("scene_a",)
    assert scenes["scene_a"].frame_names == ("frame_000000", "frame_000010", "frame_000020")

    # Stage assignments are unnecessary when selecting scenes without a stage restriction.
    (package_root / "stages/test.parquet").unlink()
    assert tuple(load_scene_metadata(package_root, "test")) == ("scene_a", "scene_b")
    with pytest.raises(FileNotFoundError):
        load_scene_metadata(package_root, "test", stages=1)


@pytest.mark.parametrize("stages", [4, "1:4", "1:1000000000000"])
def test_missing_metadata_stages_fail(package_root: Path, stages: int | str) -> None:
    """
    Reject unavailable stages, including very large requested ranges.

    Args:
        package_root: Package containing only stages 1 through 3.
        stages: Unavailable selection.
    """
    with pytest.raises(ValueError, match="not all packaged"):
        load_scene_metadata(package_root, "test", stages=stages)


@pytest.mark.parametrize("scene_ids", [[], ["scene_a", "scene_a"], ["missing"], ["scene_b"]])
def test_invalid_metadata_scene_selection_fails(package_root: Path, scene_ids: list[str]) -> None:
    """
    Reject empty, repeated, unknown, and stage-excluded scene requests.

    Args:
        package_root: Stage 1 contains scene_a only.
        scene_ids: Invalid scene subset.
    """
    with pytest.raises((ValueError, KeyError)):
        load_scene_metadata(package_root, "test", stages=1, scene_ids=scene_ids)


@pytest.mark.parametrize("package_root", ["train"], indirect=True)
def test_training_metadata_is_unstaged(package_root: Path) -> None:
    """
    Select training scenes without a stage table and reject stage arguments.

    Args:
        package_root: Training package with no stage assignments.
    """
    (package_root / "queries/train.parquet").unlink()
    assert tuple(load_scene_metadata(package_root, "train")) == ("scene_a", "scene_b")
    with pytest.raises(ValueError, match="unstaged"):
        load_scene_metadata(package_root, "train", stages=1)


def test_selected_frames_use_explicit_indices(package_root: Path) -> None:
    """
    Index reordered frame rows and limit frame-value validation to the requested scenes.

    Args:
        package_root: Package whose unselected scene will have invalid frame indices.
    """
    path = package_root / "frames/test.parquet"
    table = pq.read_table(path)
    rows = list(reversed(table.to_pylist()))
    for row in rows:
        if row["scene_id"] == "scene_b":
            row["frame_idx"] = -1
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)

    scenes = load_scene_metadata(package_root, "test", scene_ids=["scene_a"])
    assert scenes["scene_a"].frame_names == ("frame_000000", "frame_000010", "frame_000020")
    with pytest.raises(ValueError):
        load_scene_metadata(package_root, "test", scene_ids=["scene_b"])


def test_incomplete_selected_frame_mapping_fails(package_root: Path) -> None:
    """
    Reject missing frames needed to align source data with a selected scene.

    Args:
        package_root: Package to corrupt.
    """
    path = package_root / "frames/test.parquet"
    table = pq.read_table(path)
    pq.write_table(table.slice(1), path)
    with pytest.raises(ValueError, match="incomplete or ambiguous"):
        load_scene_metadata(package_root, "test", stages=1)


def test_stage_membership_audit_belongs_to_annotation_reader(package_root: Path) -> None:
    """
    Leave full query-to-stage membership checks to the reader that loads queries.

    Args:
        package_root: Package from which a stage-2 assignment will be removed.
    """
    path = package_root / "stages/test.parquet"
    table = pq.read_table(path)
    pq.write_table(table.slice(1), path)

    assert tuple(load_scene_metadata(package_root, "test", stages=1)) == ("scene_a",)
    with pytest.raises(ValueError, match="membership differs"):
        EgoRecallAnnotations(package_root)


@pytest.mark.parametrize("change", ["scene", "split", "stage_gap", "null"])
def test_invalid_stage_selection_metadata_fails(package_root: Path, change: str) -> None:
    """
    Reject unknown selected scenes, wrong splits, unavailable stages, and null selection fields.

    Args:
        package_root: Package to corrupt.
        change: Stage-table inconsistency to introduce while preserving its row count.
    """
    path = package_root / "stages/test.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    if change == "scene":
        rows[0]["scene_id"] = "unknown"
    elif change == "split":
        rows[0]["split"] = "val"
    elif change == "stage_gap":
        for row in rows:
            if row["stage"] == 2:
                row["stage"] = 3
    else:
        rows[0]["stage"] = None
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)

    with pytest.raises((ValueError, KeyError)):
        load_scene_metadata(package_root, "test", stages=2)


def test_manifest_count_audit_belongs_to_annotation_reader(package_root: Path) -> None:
    """
    Keep package-wide count checks in the full reader rather than scene preparation.

    Args:
        package_root: Package whose scene count will be changed.
    """
    path = package_root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["counts"]["scenes"] = 3
    path.write_text(json.dumps(manifest))
    assert tuple(load_scene_metadata(package_root, "test")) == ("scene_a", "scene_b")
    with pytest.raises(ValueError, match="manifest/counts/scenes"):
        EgoRecallAnnotations(package_root)

    del manifest["counts"]["frames"]
    path.write_text(json.dumps(manifest))
    assert tuple(load_scene_metadata(package_root, "test")) == ("scene_a", "scene_b")
    with pytest.raises(KeyError, match="frames"):
        EgoRecallAnnotations(package_root)
