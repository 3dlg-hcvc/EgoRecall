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
    Read the dataset manifest at EGORECALL_TEST_DATASET and open its declared split.

    Returns:
        Reader for the split declared in manifest.json.
    """
    location = os.environ.get("EGORECALL_TEST_DATASET")
    if not location:
        pytest.skip("Set EGORECALL_TEST_DATASET to check a real local annotation package.")

    root = Path(location)
    manifest = json.loads((root / "manifest.json").read_text())
    return EgoRecallAnnotations(root, split=manifest["selection"]["split"])


def test_local_queries_match_parquet(local_annotations: EgoRecallAnnotations) -> None:
    """
    Compare every returned row and stable-key lookup with the Parquet query table.

    Args:
        local_annotations: Reader for the local annotation dataset.
    """
    annotations = local_annotations
    original = pq.read_table(annotations.root / "queries" / f"{annotations.split}.parquet")
    assert annotations.query_table.equals(original)
    assert len(annotations) == original.num_rows

    for row in original.to_pylist():
        assert annotations.get_query(row["scene_id"], row["query_idx"]) == row
        assert decode_program(row["program_json"]) == json.loads(row["program_json"])


def test_local_stages_match_assignments(local_annotations: EgoRecallAnnotations) -> None:
    """
    Verify stage joins directly against the stage-assignment table.

    Args:
        local_annotations: Reader for a local dataset with stage assignments.
    """
    annotations = local_annotations
    if annotations.split == "train":
        assert annotations.available_stages == ()
        return

    assignments = pq.read_table(annotations.root / "stages" / f"{annotations.split}.parquet").to_pylist()
    for row in assignments:
        assert annotations.stage_for(row["scene_id"], row["query_idx"]) == row["stage"]
    assert {row["stage"] for row in assignments} == set(annotations.available_stages)


def test_local_annotations_and_frames_match(local_annotations: EgoRecallAnnotations) -> None:
    """
    Compare all scene annotations and frame mappings with their JSON and Parquet files,
    including objects that are not targeted by the selected queries.

    Args:
        local_annotations: Reader for the local annotation dataset.
    """
    annotations = local_annotations
    for scene_id in annotations.scene_ids:
        scene_record = annotations.get_scene(scene_id)
        with gzip.open(annotations.root / scene_record["annotations"], "rt", encoding="utf-8") as stream:
            expected = json.load(stream)
        assert annotations.get_annotations(scene_id) == expected
        assert len(expected["objects"]) == scene_record["num_objects"]

    frames = pq.read_table(annotations.root / "frames" / f"{annotations.split}.parquet").to_pylist()
    for row in frames:
        assert annotations.get_frame_name(row["scene_id"], row["frame_idx"]) == row["frame_name"]
