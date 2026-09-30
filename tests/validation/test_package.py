"""
Corrupt small annotation packages and check that validate_annotation_package() rejects each error: duplicate or
missing table rows, schema and manifest errors, missing visibility records, unsafe paths, and invalid stage tables.
"""

import gzip
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from egorecall.validation.package import validate_annotation_package


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
def test_missing_table_row_fails(package_root: Path, name: str) -> None:
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


def test_schema_error_names_type_differences(package_root: Path) -> None:
    """
    Name a column whose type differs from QUERY_SCHEMA, although its name matches.

    Args:
        package_root: Dataset whose query_idx column will be stored as int64.
    """
    path = package_root / "queries/test.parquet"
    table = pq.read_table(path)
    position = table.schema.get_field_index("query_idx")
    pq.write_table(table.set_column(position, "query_idx", table["query_idx"].cast(pa.int64())), path)

    with pytest.raises(ValueError, match="expected query_idx: int32, got query_idx: int64"):
        validate_annotation_package(package_root)


def test_invalid_query_time_fails(package_root: Path) -> None:
    """
    A query time outside its scene's sampled frames is a data error.

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
        ((), "schema_version"),
        ((), "splits"),
        ((), "counts"),
        (("splits", "test"), "counts"),
        (("splits", "test"), "stages"),
        (("splits", "test", "stages"), "first"),
        (("splits", "test", "stages"), "last"),
        (("splits", "test", "counts"), "queries"),
        (("splits", "test", "counts"), "stage_assignments"),
        (("splits", "test", "counts"), "frames"),
        (("splits", "test", "counts"), "scenes"),
        (("counts",), "queries"),
    ],
)
def test_required_manifest_fields_fail_when_missing(package_root: Path, section: tuple[str, ...], key: str) -> None:
    """
    Raise KeyError when a required manifest field is missing.

    Args:
        package_root: Synthetic package to corrupt.
        section: Keys leading from the manifest root to the object that loses a field.
        key: Required field to remove.
    """
    path = package_root / "manifest.json"
    manifest = json.loads(path.read_text())
    record = manifest
    for name in section:
        record = record[name]
    del record[key]
    path.write_text(json.dumps(manifest))

    with pytest.raises(KeyError, match=key):
        validate_annotation_package(package_root)


def test_unsupported_manifest_schema_fails(package_root: Path) -> None:
    """
    Reject a manifest whose schema_version is not 2, the only supported layout.

    Args:
        package_root: Package whose manifest will declare schema version 1.
    """
    path = package_root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["schema_version"] = 1
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="Unsupported package schema_version: 1"):
        validate_annotation_package(package_root)


@pytest.mark.parametrize("section", ["splits", "counts"])
@pytest.mark.parametrize("value", [None, []])
def test_malformed_manifest_records_fail(package_root: Path, section: str, value: list[object] | None) -> None:
    """
    Reject splits and counts sections that are not JSON objects.

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
    Reject a manifest that does not list the split assigned to scenes in scenes.json.

    Args:
        package_root: Test package whose manifest will list val instead of test.
    """
    path = package_root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["splits"] = {"val": manifest["splits"]["test"]}
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="is not listed in manifest.json"):
        validate_annotation_package(package_root)


def test_manifest_counts_require_integers(package_root: Path) -> None:
    """
    Require integer types for manifest counts.

    Args:
        package_root: Package whose query count will have the wrong type.
    """
    path = package_root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["counts"]["queries"] = 4.0
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="manifest/counts/queries"):
        validate_annotation_package(package_root)


@pytest.mark.parametrize("change", ["scene", "split", "stage_gap", "null"])
def test_invalid_stage_table_fails(package_root: Path, change: str) -> None:
    """
    Reject unknown scenes, wrong splits, unavailable stages, and null fields in the stage table.

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
        validate_annotation_package(package_root)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("scene_id", "scene a", "Invalid scene ID"),
        ("source_fps", 0.0, "positive frame rate"),
        ("nominal_timeline_fps", 0, "positive frame rate"),
    ],
)
def test_unusable_scene_ids_and_rates_fail(package_root: Path, field: str, value: object, message: str) -> None:
    """
    Reject scene IDs that are not single directory names and zero frame rates, which preparation cannot use.

    Args:
        package_root: Package whose first scene record will be changed.
        field: scenes.json field to change.
        value: Invalid replacement value.
        message: Expected error text.
    """
    path = package_root / "scenes.json"
    scenes = json.loads(path.read_text())
    scenes[0][field] = value
    path.write_text(json.dumps(scenes))

    with pytest.raises(ValueError, match=message):
        validate_annotation_package(package_root)


@pytest.mark.parametrize(("change", "message"), [("form", "Invalid ScanNet"), ("order", "must increase")])
def test_frame_names_must_be_increasing_source_names(package_root: Path, change: str, message: str) -> None:
    """
    Reject frame names that are not ScanNet++ frame names or do not increase with frame_idx,
    since preparation extracts source frames in video order.

    Args:
        package_root: Package whose first scene's frame names will be changed.
        change: Replace one name with another form, or swap two names.
        message: Expected error text.
    """
    path = package_root / "frames/test.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    if change == "form":
        rows[1]["frame_name"] = "image_000010"
    else:
        rows[0]["frame_name"], rows[1]["frame_name"] = rows[1]["frame_name"], rows[0]["frame_name"]
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)

    with pytest.raises(ValueError, match=message):
        validate_annotation_package(package_root)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("emit_reason", "unknown emit_reason"),
        ("duplicate_target", "duplicate object IDs in target_oids"),
        ("overlapping_targets", "do not partition the target IDs"),
        ("program_operand", "invalid program_json"),
        ("program_text", "invalid program_json"),
        ("description", "description: expected a nonempty string"),
    ],
)
def test_invalid_query_records_fail(package_root: Path, change: str, message: str) -> None:
    """
    Reject query rows with an unknown emission reason, repeated or overlapping target IDs, a malformed
    program, or a blank description, although the table schema is correct.

    Args:
        package_root: Package whose first query row will be changed.
        change: Invalid value to introduce.
        message: Expected error text.
    """
    path = package_root / "queries/test.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    query = rows[0]
    if change == "emit_reason":
        query["emit_reason"] = "repeat"
    elif change == "duplicate_target":
        query["target_oids"] = [1, 1]
    elif change == "overlapping_targets":
        query["hidden_target_oids"] = [1]
    elif change == "program_operand":
        query["program_json"] = json.dumps(["first_seen", True])
    elif change == "program_text":
        query["program_json"] = "first_seen chair"
    else:
        query["description"] = "   "
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)

    with pytest.raises(ValueError, match=message):
        validate_annotation_package(package_root)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("schema_version", "unsupported annotation schema_version"),
        ("num_frames", "annotation scene or timeline does not match"),
        ("object_count", "object count does not match"),
        ("object_id", "invalid annotation object ID"),
        ("missing_field", "expected fields"),
        ("summary_frame", "frame outside the sampled frame sequence"),
        ("visible_frames", "exceeds the scene length"),
        ("segment", "invalid visibility segment bounds"),
        ("observation_frame", "observation frame outside"),
        ("fraction", "finite, nonnegative number"),
    ],
)
def test_invalid_scene_annotations_fail(package_root: Path, change: str, message: str) -> None:
    """
    Reject visibility files whose structure, object IDs, or frame references disagree with scenes.json
    or fall outside the scene's sampled frames.

    Args:
        package_root: Package whose scene_a visibility file will be changed.
        change: Invalid value to introduce.
        message: Expected error text.
    """
    path = package_root / "annotations/scene_a.json.gz"
    scene_annotations = json.loads(gzip.decompress(path.read_bytes()))
    chair = scene_annotations["objects"]["1"]
    if change == "schema_version":
        scene_annotations["schema_version"] = 2
    elif change == "num_frames":
        scene_annotations["num_frames"] = 4
    elif change == "object_count":
        scene_annotations["objects"]["3"] = chair
    elif change == "object_id":
        scene_annotations["objects"]["01"] = scene_annotations["objects"].pop("1")
    elif change == "missing_field":
        del chair["label"]
    elif change == "summary_frame":
        chair["temporal"]["last_seen_frame"] = 3
    elif change == "visible_frames":
        chair["temporal"]["total_visible_frames"] = 4
    elif change == "segment":
        chair["visibility_segments"] = [[1, 0]]
    elif change == "observation_frame":
        chair["per_frame"]["3"] = chair["per_frame"]["0"]
    else:
        chair["per_frame"]["0"]["visible_area_frac"] = -0.1
    path.write_bytes(gzip.compress(json.dumps(scene_annotations).encode()))

    with pytest.raises(ValueError, match=message):
        validate_annotation_package(package_root)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("query_count", "row counts disagree with scenes.json"),
        ("duplicate_scene", "Duplicate scene metadata"),
        ("unknown_split", "Unknown split 'dev'"),
    ],
)
def test_inconsistent_package_structure_fails(package_root: Path, change: str, message: str) -> None:
    """
    Reject a scene query count that disagrees with the query table, a repeated scene record,
    and a manifest split other than train, val, or test.

    Args:
        package_root: Package whose scenes.json or manifest.json will be changed.
        change: Inconsistency to introduce.
        message: Expected error text.
    """
    if change == "unknown_split":
        path = package_root / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["splits"]["dev"] = manifest["splits"]["test"]
        path.write_text(json.dumps(manifest))
    else:
        path = package_root / "scenes.json"
        scenes = json.loads(path.read_text())
        if change == "query_count":
            scenes[0]["num_queries"] = 2
        else:
            scenes.append(scenes[0])
        path.write_text(json.dumps(scenes))

    with pytest.raises(ValueError, match=message):
        validate_annotation_package(package_root)
