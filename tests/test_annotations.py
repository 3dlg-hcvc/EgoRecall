"""
Check sparse identity, exact-stage selection, and invalid data joins using
small packages whose expected selections can be inspected manually.
"""

import gzip
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from egorecall.data import EgoRecallAnnotations, decode_program, parse_stages
from egorecall.data.validate_package import validate_annotation_package


def test_stable_keys_and_stage_selection(package_root: Path) -> None:
    """
    Join stage assignments using the scene/query ID pair across differing table orders.

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
    Report missing split membership before attempting nonexistent table paths.

    Args:
        package_root: A test-only package.
    """
    with pytest.raises(FileNotFoundError):
        EgoRecallAnnotations(package_root, split="val")


@pytest.mark.parametrize("name", ["queries", "stages", "frames"])
def test_duplicate_keys_fail(package_root: Path, name: str) -> None:
    """
    Reject duplicate identifiers in the query, stage, and frame tables.

    Args:
        package_root: Synthetic package to corrupt.
        name: Table in which to introduce a duplicate row.
    """
    path = package_root / name / "test.parquet"
    table = pq.read_table(path)
    pq.write_table(pa.concat_tables([table, table.slice(0, 1)]), path)

    with pytest.raises(ValueError, match="[Dd]uplicate"):
        validate_annotation_package(package_root)


@pytest.mark.parametrize("name", ["stages", "frames"])
def test_missing_join_row_fails(package_root: Path, name: str) -> None:
    """
    The checker detects a query without a stage assignment or a scene with a missing frame.

    Args:
        package_root: Synthetic package to corrupt.
        name: Table from which one row will be removed.
    """
    path = package_root / name / "test.parquet"
    table = pq.read_table(path)
    pq.write_table(table.slice(1), path)

    with pytest.raises(ValueError, match="membership differs|incomplete"):
        validate_annotation_package(package_root)


def test_wrong_query_schema_fails(package_root: Path) -> None:
    """
    Query-table columns must match the names specified by QUERY_SCHEMA.

    Args:
        package_root: Dataset whose source_query_id column will be given an unexpected name.
    """
    path = package_root / "queries/test.parquet"
    table = pq.read_table(path)
    names = ["instance_id" if name == "source_query_id" else name for name in table.column_names]
    pq.write_table(table.rename_columns(names), path)

    with pytest.raises(ValueError, match="incompatible schema"):
        validate_annotation_package(package_root)


def test_invalid_query_time_fails(package_root: Path) -> None:
    """
    A query outside its scene's canonical timeline is a data error.

    Args:
        package_root: Synthetic package to corrupt.
    """
    path = package_root / "queries/test.parquet"
    table = pq.read_table(path)
    records = table.to_pylist()
    records[0]["frame"] = 3
    pq.write_table(pa.Table.from_pylist(records, schema=table.schema), path)

    with pytest.raises(ValueError, match="outside the scene frame range"):
        validate_annotation_package(package_root)


def test_missing_target_annotation_fails_in_checker(package_root: Path) -> None:
    """
    The checker rejects target IDs absent from the object visibility file.

    Args:
        package_root: Package whose object 1 will be replaced by another ID.
    """
    path = package_root / "annotations/scene_a.json.gz"
    scene_annotations = json.loads(gzip.decompress(path.read_bytes()))
    scene_annotations["objects"]["3"] = scene_annotations["objects"].pop("1")
    path.write_bytes(gzip.compress(json.dumps(scene_annotations).encode()))

    with pytest.raises(ValueError, match="target IDs have no annotation"):
        validate_annotation_package(package_root)


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


def test_annotation_path_cannot_escape_package(package_root: Path) -> None:
    """
    Metadata references must use package-relative paths without traversal.

    Args:
        package_root: Package whose annotation path will be changed.
    """
    path = package_root / "scenes.json"
    scenes = json.loads(path.read_text())
    scenes[0]["annotations"] = "../outside.json.gz"
    path.write_text(json.dumps(scenes))

    with pytest.raises(ValueError, match="relative file path"):
        validate_annotation_package(package_root)


@pytest.mark.parametrize(
    ("section", "key"),
    [
        (None, "schema_version"),
        (None, "selection"),
        (None, "counts"),
        ("selection", "split"),
        ("selection", "stage_from"),
        ("selection", "stage_to"),
        ("counts", "queries"),
        ("counts", "stage_assignments"),
        ("counts", "frames"),
        ("counts", "scenes"),
    ],
)
def test_required_manifest_fields_fail_when_missing(package_root: Path, section: str | None, key: str) -> None:
    """
    Raise KeyError when a required manifest field is missing.

    Args:
        package_root: Synthetic package to corrupt.
        section: Nested manifest object, or None for the root object.
        key: Required field to remove.
    """
    path = package_root / "manifest.json"
    manifest = json.loads(path.read_text())
    record = manifest if section is None else manifest[section]
    del record[key]
    path.write_text(json.dumps(manifest))

    with pytest.raises(KeyError, match=key):
        validate_annotation_package(package_root)


@pytest.mark.parametrize("section", ["selection", "counts"])
@pytest.mark.parametrize("value", [None, []])
def test_malformed_manifest_records_fail(package_root: Path, section: str, value: list[object] | None) -> None:
    """
    Reject selection and counts sections that are not JSON objects.

    Args:
        package_root: Synthetic package to corrupt.
        section: Manifest object to replace.
        value: Deliberately invalid record value.
    """
    path = package_root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest[section] = value
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=f"manifest/{section}"):
        validate_annotation_package(package_root)


def test_manifest_split_mismatch_fails(package_root: Path) -> None:
    """
    Reject a manifest split that disagrees with the requested split.

    Args:
        package_root: Test package whose manifest will incorrectly declare val.
    """
    path = package_root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["selection"]["split"] = "val"
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="Manifest selection does not match"):
        validate_annotation_package(package_root)


def test_manifest_counts_require_integers(package_root: Path) -> None:
    """
    Require integer types for manifest population counts.

    Args:
        package_root: Package whose query count will have the wrong type.
    """
    path = package_root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["counts"]["queries"] = 4.0
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="manifest/counts/queries"):
        validate_annotation_package(package_root)


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
