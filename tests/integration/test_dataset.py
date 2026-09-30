"""
Integration checks against a local EgoRecall dataset. Set
EGORECALL_TEST_DATASET to enable them without downloading gated data in tests.
"""

import gzip
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from egorecall.data import EgoRecallAnnotations, decode_program

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def local_annotations() -> EgoRecallAnnotations:
    """
    Open the dataset at EGORECALL_TEST_DATASET for integration checks. Prefer the
    test split when available; otherwise use the first split listed in the manifest.

    Returns:
        Annotation reader for the selected split.
    """
    location = os.environ.get("EGORECALL_TEST_DATASET")
    if not location:
        pytest.skip("Set EGORECALL_TEST_DATASET to check a real local annotation package.")

    root = Path(location)
    manifest = json.loads((root / "manifest.json").read_text())
    split = "test" if "test" in manifest["splits"] else next(iter(manifest["splits"]))
    return EgoRecallAnnotations(root, split=split)


def test_local_queries_match_parquet(local_annotations: EgoRecallAnnotations) -> None:
    """
    Compare the complete query table and sample stable-key lookups across its rows.

    Args:
        local_annotations: Reader for the local annotation dataset.
    """
    annotation_reader = local_annotations
    query_table = pq.read_table(annotation_reader.root / "queries" / f"{annotation_reader.split}.parquet")
    assert annotation_reader.query_table.equals(query_table)
    assert len(annotation_reader) == query_table.num_rows

    positions = sorted(set(range(0, len(query_table), max(1, len(query_table) // 1024))) | {len(query_table) - 1})
    for row in query_table.take(positions).to_pylist():
        assert annotation_reader.get_query(row["scene_id"], row["query_idx"]) == row
        assert decode_program(row["program_json"]) == json.loads(row["program_json"])


def test_local_stages_match_assignments(local_annotations: EgoRecallAnnotations) -> None:
    """
    Check each query's stage against its row in the stage-assignment table.

    Args:
        local_annotations: Reader for a local dataset with stage assignments.
    """
    annotation_reader = local_annotations
    if annotation_reader.split == "train":
        assert annotation_reader.available_stages == ()
        return

    assignments = pq.read_table(annotation_reader.root / "stages" / f"{annotation_reader.split}.parquet").to_pylist()
    for row in assignments:
        assert annotation_reader.stage_for(row["scene_id"], row["query_idx"]) == row["stage"]
    assert {row["stage"] for row in assignments} == set(annotation_reader.available_stages)


def test_local_annotations_and_frames_match(local_annotations: EgoRecallAnnotations) -> None:
    """
    Compare all scene annotations and frame mappings with their JSON and Parquet files,
    including objects that are not targeted by the selected queries.

    Args:
        local_annotations: Reader for the local annotation dataset.
    """
    annotation_reader = local_annotations
    for scene_id in annotation_reader.scene_ids:
        scene_record = annotation_reader.get_scene(scene_id)
        with gzip.open(annotation_reader.root / scene_record["annotations"], "rt", encoding="utf-8") as stream:
            expected = json.load(stream)
        assert annotation_reader.get_annotations(scene_id) == expected
        assert len(expected["objects"]) == scene_record["num_objects"]

    frames = pq.read_table(annotation_reader.root / "frames" / f"{annotation_reader.split}.parquet").to_pylist()
    for row in frames:
        assert annotation_reader.get_frame_name(row["scene_id"], row["frame_idx"]) == row["frame_name"]
