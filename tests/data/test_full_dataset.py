"""
Read and check multi-split datasets using small, directly constructed tables.
"""

import gzip
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from egorecall import DatasetPaths
from egorecall.data import EgoRecallAnnotations
from egorecall.validation.check import check_dataset
from egorecall.validation.package import validate_annotation_package
from tests.helpers import _add_manifest_hashes


@pytest.fixture
def full_package(package_root: Path) -> Path:
    """
    Extend the four-query test fixture with independent train and validation scenes.
    Sparse query IDs recur across splits to check selection and scene-scoped lookups.

    Args:
        package_root: Two-scene test dataset with three stages.

    Returns:
        Three-split dataset with a schema-2 manifest and unstaged training.
    """
    scene_records = json.loads((package_root / "scenes.json").read_text())
    all_scene_records = list(scene_records)
    test_counts = json.loads((package_root / "manifest.json").read_text())["splits"]["test"]["counts"]
    splits = {"test": {"counts": test_counts, "stages": {"first": 1, "last": 3}}}

    # Use distinct scene IDs in each split while preserving the query and frame indices.
    for split in ("train", "val"):
        for table_name in ("queries", "frames", "stages"):
            if split == "train" and table_name == "stages":
                continue
            table = pq.read_table(package_root / table_name / "test.parquet")
            rows = table.to_pylist()
            for row in rows:
                scene_id = row["scene_id"]
                row["scene_id"] = f"{split}_{scene_id}"
                if table_name != "frames":
                    row["split"] = split
            pq.write_table(
                pa.Table.from_pylist(rows, schema=table.schema), package_root / table_name / f"{split}.parquet"
            )

        for scene_record in scene_records:
            original_scene_id = scene_record["scene_id"]
            scene_id = f"{split}_{original_scene_id}"
            annotation_path = f"annotations/{scene_id}.json.gz"
            with gzip.open(package_root / scene_record["annotations"], "rt") as stream:
                scene_annotations = json.load(stream)
            scene_annotations["scene_id"] = scene_id
            (package_root / annotation_path).write_bytes(gzip.compress(json.dumps(scene_annotations).encode()))
            all_scene_records.append(
                {**scene_record, "scene_id": scene_id, "split": split, "annotations": annotation_path}
            )

        counts = {**test_counts, "stage_assignments": 0 if split == "train" else 4}
        splits[split] = {"counts": counts, "stages": None if split == "train" else {"first": 1, "last": 3}}

    # Explicit totals let the checker tests detect disagreements between split and dataset counts.
    manifest = {
        "schema_version": 2,
        "dataset_id": "example/EgoRecall",
        "dataset_version": "1.0.0",
        "splits": splits,
        "counts": dict(queries=12, stage_assignments=8, scenes=6, frames=18, objects=12, any_target_queries=0),
    }
    (package_root / "scenes.json").write_text(json.dumps(all_scene_records))
    (package_root / "manifest.json").write_text(json.dumps(manifest))
    _add_manifest_hashes(package_root)
    return package_root


def test_splits_have_independent_queries_and_stages(full_package: Path) -> None:
    """
    Read all three splits and keep training unstaged, even when query IDs recur.

    Args:
        full_package: Dataset with two scenes and four queries per split.
    """
    for split in ("train", "val", "test"):
        annotation_reader = EgoRecallAnnotations(full_package, split=split)
        scene_id = "scene_a" if split == "test" else f"{split}_scene_a"
        assert annotation_reader.available_splits == ("train", "val", "test")
        assert len(annotation_reader) == 4
        assert annotation_reader.get_query(scene_id, 103)["split"] == split
        assert annotation_reader.stage_for(scene_id, 103) == (None if split == "train" else 3)
        assert set(annotation_reader.get_annotations(scene_id)["objects"]) == {"1", "2"}
    report = check_dataset(DatasetPaths(full_package))
    assert report.queries_checked == 12
    assert report.annotation_scenes == 6


def test_stage_filtering_reads_only_selected_queries(full_package: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Read only selected query rows while retaining their IDs and the scene's full frame mapping.

    Args:
        full_package: Dataset with three evaluation stages.
        monkeypatch: Fixture recording query rows returned by Parquet.
    """
    read_table = pq.read_table
    query_reads = []

    def recorded_read(path: Path, **kwargs: object) -> pa.Table:
        """
        Record query row counts without changing Parquet behavior.

        Args:
            path: Table filename.
            kwargs: Parquet read options.

        Returns:
            Rows returned by pq.read_table().
        """
        table = read_table(path, **kwargs)
        if Path(path).parent.name == "queries":
            query_reads.append(table.num_rows)
        return table

    monkeypatch.setattr(pq, "read_table", recorded_read)
    annotation_reader = EgoRecallAnnotations(full_package, split="test", stages=3)
    assert annotation_reader.query_keys == (("scene_a", 103),)
    assert annotation_reader.available_stages == (1, 2, 3)
    assert query_reads == [1]
    assert annotation_reader.stage_for("scene_a", 103) == 3
    assert len(annotation_reader.frame_names("scene_a")) == 3


def test_aggregate_counts_are_checked(full_package: Path) -> None:
    """
    Reject inconsistent totals even when each split is internally consistent.

    Args:
        full_package: Dataset whose aggregate query count will be changed.
    """
    path = full_package / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["counts"]["queries"] += 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest/counts/queries"):
        validate_annotation_package(full_package)


@pytest.mark.parametrize("change", ["missing_split", "training_stages", "invalid_stage_bounds", "invalid_total_counts"])
def test_invalid_full_manifest_is_rejected(full_package: Path, change: str) -> None:
    """
    Detect missing splits, invalid stage ranges, and malformed aggregate counts.

    Args:
        full_package: Three-split dataset to alter.
        change: Invalid manifest declaration to introduce.
    """
    path = full_package / "manifest.json"
    manifest = json.loads(path.read_text())
    if change == "missing_split":
        del manifest["splits"]["val"]
    elif change == "training_stages":
        manifest["splits"]["train"]["stages"] = {"first": 1, "last": 3}
    elif change == "invalid_stage_bounds":
        manifest["splits"]["test"]["stages"] = None
    else:
        manifest["counts"] = None
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        validate_annotation_package(full_package)
