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
