"""
Read queries by their stable keys, select stages and scenes, and map frame numbers to source filenames,
using small packages whose expected selections can be inspected manually.
"""

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from egorecall.data import EgoRecallAnnotations, decode_program, parse_stages
from egorecall.data.annotations import select_scene_ids
from egorecall.validation.package import validate_annotation_package


def test_stable_keys_and_stage_selection(package_root: Path) -> None:
    """
    Match stage assignments to queries by the scene/query ID pair across differing table orders.

    Args:
        package_root: Synthetic package with three stages and two scenes.
    """
    third = EgoRecallAnnotations(package_root, stages=3)
    assert third.query_keys == (("scene_a", 103),)
    assert third.stage_for("scene_a", 103) == 3
    assert third.get_query("scene_a", 103)["frame"] == 2

    with pytest.raises(KeyError):
        third.get_query("scene_a", 0)
    with pytest.raises(KeyError):
        third.get_query("scene_a", 4)

    middle = EgoRecallAnnotations(package_root, stages="2:3")
    assert middle.query_keys == (("scene_a", 17), ("scene_a", 103), ("scene_b", 4))
    assert middle.get_query("scene_b", 4)["scene_id"] == "scene_b"
    assert middle.available_stages == (1, 2, 3)
    assert middle.get_scene("scene_a")["num_queries"] == 3
    assert list(middle.iter_queries(batch_size=1)) == middle.query_table.to_pylist()

    # Selecting the full stage range preserves the complete query table and its order.
    all_queries = EgoRecallAnnotations(package_root)
    all_stages = EgoRecallAnnotations(package_root, stages="1:3")
    assert all_queries.query_keys == (("scene_a", 4), ("scene_a", 17), ("scene_a", 103), ("scene_b", 4))
    assert all_stages.query_keys == all_queries.query_keys
    assert all_stages.query_table.equals(all_queries.query_table)
    assert [all_stages.stage_for(*key) for key in all_stages.query_keys] == [1, 2, 3, 2]


def test_full_scene_context_and_fresh_records(package_root: Path) -> None:
    """
    Retain contextual annotations and keep query and scene lookups unchanged
    after a caller edits a returned dictionary.

    Args:
        package_root: Synthetic package including an untargeted table object.
    """
    annotation_reader = EgoRecallAnnotations(package_root, stages=1)
    query = annotation_reader.get_query("scene_a", 4)
    assert decode_program(query["program_json"]) == ["first_seen", "chair"]

    query["target_oids"].append(999)
    assert annotation_reader.get_query("scene_a", 4)["target_oids"] == [1]

    # Changes to returned metadata must not affect later annotation or frame lookups.
    scene_record = annotation_reader.get_scene("scene_a")
    scene_record["num_frames"] = 999
    scene_record["annotations"] = "annotations/nonexistent.json.gz"
    assert annotation_reader.get_scene("scene_a")["num_frames"] == 3

    scene_annotations = annotation_reader.get_annotations("scene_a")
    assert scene_annotations["objects"]["2"]["label"] == "table"
    assert scene_annotations["objects"]["1"]["visibility_segments"] == [[0, 1]]
    assert annotation_reader.frame_names("scene_a") == ("frame_000000", "frame_000010", "frame_000020")


@pytest.mark.parametrize("scene_id", ["scene_b", "unknown_scene"])
def test_scene_access_requires_selection(package_root: Path, scene_id: str) -> None:
    """
    Reject both existing but unselected scenes and unknown scenes across scene accessors.

    Args:
        package_root: Synthetic package with only scene_a represented in stage 1.
        scene_id: Scene outside the reader's selection.
    """
    annotation_reader = EgoRecallAnnotations(package_root, stages=1)
    with pytest.raises(KeyError):
        annotation_reader.get_scene(scene_id)
    with pytest.raises(KeyError):
        annotation_reader.frame_names(scene_id)
    with pytest.raises(KeyError):
        annotation_reader.get_frame_name(scene_id, 0)
    with pytest.raises(KeyError):
        annotation_reader.get_annotations(scene_id)


def test_reordered_frame_rows_use_explicit_indices(package_root: Path) -> None:
    """
    Reordered frame rows retain the mapping from sampled frame numbers to source filenames.

    Args:
        package_root: Synthetic package whose frame rows will be reversed.
    """
    path = package_root / "frames/test.parquet"
    table = pq.read_table(path)
    pq.write_table(table.take(list(reversed(range(len(table))))), path)

    annotation_reader = EgoRecallAnnotations(package_root)
    assert annotation_reader.get_frame_name("scene_a", 2) == "frame_000020"

    assert annotation_reader.get_frame_name("scene_a", -1) == "frame_000020"
    with pytest.raises(IndexError):
        annotation_reader.get_frame_name("scene_a", 3)


@pytest.mark.parametrize("stages", [4, "1:4", "1:1000000000000"])
def test_unavailable_stage_range_fails(package_root: Path, stages: int | str) -> None:
    """
    Raise when requested stages are absent from the assignment table.

    Args:
        package_root: Package containing stages 1 through 3 only.
        stages: Selection extending beyond the packaged stages.
    """
    with pytest.raises(ValueError, match="not all packaged"):
        EgoRecallAnnotations(package_root, stages=stages)


@pytest.mark.parametrize("value", [0, -1, True, "0", "3:1", "1:", "1:2:3", "all", "1.5"])
def test_invalid_stage_specification(value: int | str) -> None:
    """
    Reject ambiguous, nonpositive, and reversed stage specifications.

    Args:
        value: Invalid stage expression.
    """
    with pytest.raises(ValueError):
        parse_stages(value)


@pytest.mark.parametrize("package_root", ["train"], indirect=True)
def test_training_is_unstaged(package_root: Path) -> None:
    """
    A training package works without assignments and rejects stage selection.

    Args:
        package_root: Synthetic training package with no stage table.
    """
    annotation_reader = EgoRecallAnnotations(package_root, split="train")
    assert len(annotation_reader) == 4
    assert annotation_reader.available_stages == ()
    assert annotation_reader.stage_for("scene_a", 4) is None

    with pytest.raises(ValueError, match="unstaged"):
        EgoRecallAnnotations(package_root, split="train", stages=1)


def test_unavailable_split_fails(package_root: Path) -> None:
    """
    Report a split absent from the dataset directory before reading its nonexistent tables.

    Args:
        package_root: A test-only package.
    """
    with pytest.raises(ValueError, match="not in this dataset directory"):
        EgoRecallAnnotations(package_root, split="val")


def test_symlinked_payloads_work(package_root: Path, tmp_path: Path) -> None:
    """
    Cached downloads may link package files to a shared external blob store.

    Args:
        package_root: Synthetic package with a payload to replace by a symlink.
        tmp_path: Directory holding the external payload.
    """
    payload = package_root / "queries/test.parquet"
    blob = tmp_path / "cached_blob"
    payload.rename(blob)
    payload.symlink_to(blob)
    assert len(EgoRecallAnnotations(package_root)) == 4


def test_readers_do_not_repeat_package_audit(package_root: Path) -> None:
    """
    Manifest totals are checked explicitly rather than whenever query rows are loaded.

    Args:
        package_root: Valid query files with a deliberately incorrect manifest total.
    """
    manifest_path = package_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["counts"]["queries"] = 999
    manifest_path.write_text(json.dumps(manifest))
    annotation_reader = EgoRecallAnnotations(package_root)
    assert len(annotation_reader) == 4
    assert annotation_reader.get_query("scene_a", 4)["target_oids"] == [1]
    with pytest.raises(ValueError, match="manifest/counts/queries"):
        validate_annotation_package(package_root)


def test_scene_selection_keeps_requested_order(package_root: Path) -> None:
    """
    Select requested scenes from a stage selection in the caller's order, or all selected scenes when omitted.

    Args:
        package_root: Package whose stage 2 contains scene_a and scene_b.
    """
    annotation_reader = EgoRecallAnnotations(package_root, stages=2)
    assert select_scene_ids(annotation_reader.scene_ids, None) == ("scene_a", "scene_b")
    assert select_scene_ids(annotation_reader.scene_ids, ["scene_b", "scene_a"]) == ("scene_b", "scene_a")


@pytest.mark.parametrize("scene_ids", [[], ["scene_a", "scene_a"], ["missing"], ["scene_b"]])
def test_invalid_scene_selection_fails(package_root: Path, scene_ids: list[str]) -> None:
    """
    Reject empty, repeated, unknown, and stage-excluded scene requests.

    Args:
        package_root: Package whose stage 1 contains scene_a only.
        scene_ids: Invalid scene subset.
    """
    annotation_reader = EgoRecallAnnotations(package_root, stages=1)
    with pytest.raises((ValueError, KeyError)):
        select_scene_ids(annotation_reader.scene_ids, scene_ids)


def test_incomplete_frame_mapping_fails(package_root: Path) -> None:
    """
    Fail when a selected scene lacks a frame row, instead of pairing queries with the wrong images.

    Args:
        package_root: Package whose first frame row will be removed.
    """
    path = package_root / "frames/test.parquet"
    table = pq.read_table(path)
    pq.write_table(table.slice(1), path)
    with pytest.raises(KeyError):
        EgoRecallAnnotations(package_root, stages=1)
