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

from egorecall.data import EgoRecallDataset, decode_program

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def local_dataset() -> EgoRecallDataset:
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
    return EgoRecallDataset(root, split=manifest["selection"]["split"])


def test_local_queries_match_parquet(local_dataset: EgoRecallDataset) -> None:
    """
    Compare every returned row and stable-key lookup with the Parquet query table.

    Args:
        local_dataset: Reader for the local annotation dataset.
    """
    dataset = local_dataset
    original = pq.read_table(dataset.root / "queries" / f"{dataset.split}.parquet")
    assert dataset.query_table.equals(original)
    assert len(dataset) == original.num_rows
    for row in original.to_pylist():
        assert dataset.get_query(row["scene_id"], row["query_idx"]) == row
        assert decode_program(row["program_json"]) == json.loads(row["program_json"])


def test_local_stages_match_assignments(local_dataset: EgoRecallDataset) -> None:
    """
    Verify stage joins directly against the stage-assignment table.

    Args:
        local_dataset: Reader for a local dataset with stage assignments.
    """
    dataset = local_dataset
    if dataset.split == "train":
        assert dataset.available_stages == ()
        return
    assignments = pq.read_table(dataset.root / "stages" / f"{dataset.split}.parquet").to_pylist()
    for row in assignments:
        assert dataset.stage_for(row["scene_id"], row["query_idx"]) == row["stage"]
    assert {row["stage"] for row in assignments} == set(dataset.available_stages)


def test_local_annotations_and_frames_match(local_dataset: EgoRecallDataset) -> None:
    """
    Compare all scene annotations and frame mappings with their JSON and Parquet files,
    including objects that are not targeted by the selected queries.

    Args:
        local_dataset: Reader for the local annotation dataset.
    """
    dataset = local_dataset
    for scene_id in dataset.scene_ids:
        scene = dataset.get_scene(scene_id)
        with gzip.open(dataset.root / scene["annotations"], "rt", encoding="utf-8") as stream:
            expected = json.load(stream)
        assert dataset.get_annotations(scene_id) == expected
        assert len(expected["objects"]) == scene["num_objects"]
    frames = pq.read_table(dataset.root / "frames" / f"{dataset.split}.parquet").to_pylist()
    for row in frames:
        assert dataset.get_frame_name(row["scene_id"], row["frame_idx"]) == row["frame_name"]
