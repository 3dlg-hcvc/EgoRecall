"""
Read scene settings and frame names without loading query text or object visibility.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq

from egorecall.data.records import SceneRecord
from egorecall.data.stages import StageRange, parse_stages, require_stage_range


@dataclass(frozen=True)
class SceneMetadata:
    """
    The scene's settings from scenes.json and the source name of each sampled frame.

    Args:
        scene_record: Counts, sampling stride, frame rate, and annotation filename.
        frame_names: Source names ordered by frame_idx from frames/<split>.parquet.
    """

    scene_record: SceneRecord
    frame_names: tuple[str, ...]


def select_scene_ids(available: tuple[str, ...], requested: list[str] | None) -> tuple[str, ...]:
    """
    Select scene IDs while preserving the order requested by the caller.

    Args:
        available: IDs represented in the selected split and stages.
        requested: IDs to use, or None for every available scene.

    Returns:
        The selected IDs. A requested ID absent from available raises KeyError.
    """
    if requested is None:
        return available

    if not requested or len(requested) != len(set(requested)):
        raise ValueError("Scene selection must be nonempty and contain no duplicates.")

    missing = set(requested) - set(available)
    if missing:
        raise KeyError(f"Scenes are absent from the requested split/stage selection: {sorted(missing)}.")
    return tuple(requested)


def index_frame_names(frame_table: pa.Table, scene_records: dict[str, SceneRecord]) -> dict[str, tuple[str, ...]]:
    """
    Build a filename lookup for each scene. Frame rows may be stored out of order,
    so use frame_idx to put them in image order. For example, frame_idx 1 may map
    to frame_000010 when every tenth ScanNet++ frame is sampled.

    Args:
        frame_table: scene_id, frame_idx, and frame_name columns from frames/<split>.parquet.
        scene_records: Scene records whose num_frames determines each lookup's length.

    Returns:
        Tuples keyed by scene ID; indexing a tuple by a query's frame gives its source filename.
    """
    names_by_scene: dict[str, dict[int, str]] = {scene_id: {} for scene_id in scene_records}
    for batch in frame_table.to_batches(max_chunksize=8192):
        for frame_row in batch.to_pylist():
            names_by_scene[frame_row["scene_id"]][frame_row["frame_idx"]] = frame_row["frame_name"]
    return {
        scene_id: tuple(names_by_scene[scene_id][frame_idx] for frame_idx in range(scene_record["num_frames"]))
        for scene_id, scene_record in scene_records.items()
    }


def read_scene_records(dataset_root: Path, split: str) -> dict[str, SceneRecord]:
    """
    Read sampling settings and counts for the scenes in one split.

    Args:
        dataset_root: Directory containing scenes.json.
        split: Benchmark split to select.

    Returns:
        Records from scenes.json keyed by scene ID.
    """
    with (dataset_root / "scenes.json").open(encoding="utf-8") as stream:
        scene_records = cast(list[SceneRecord], json.load(stream))
    return {scene_record["scene_id"]: scene_record for scene_record in scene_records if scene_record["split"] == split}


def load_scene_metadata(
    dataset_root: Path,
    split: str,
    *,
    stages: int | str | None = None,
    scene_ids: list[str] | None = None,
) -> dict[str, SceneMetadata]:
    """
    Read settings and frame names for the scenes to prepare. Stage assignments
    determine which scenes to include; every included scene keeps all its frames.
    Run egorecall-check first to validate the annotation files.

    Args:
        dataset_root: Directory containing scenes.json and the frame and stage tables.
        split: Benchmark split to read.
        stages: Exact stage, inclusive range, or None for all scenes. Omit for training.
        scene_ids: Optional scene subset, preserving the supplied order.

    Returns:
        Scene settings and complete frame-name sequences keyed by scene ID.
    """
    dataset_root = dataset_root.expanduser().resolve()

    if split not in ("train", "val", "test"):
        raise ValueError(f"Unknown split {split!r}; use train, val, or test.")

    stage_range = parse_stages(stages) if stages is not None else None
    if split == "train" and stage_range is not None:
        raise ValueError("Training is unstaged; omit stages when reading train.")

    scene_records = read_scene_records(dataset_root, split)
    available = tuple(
        sorted(scene_id for scene_id, scene_record in scene_records.items() if scene_record["num_queries"])
    )
    if stage_range is not None:
        available = _stage_scene_ids(dataset_root, split, stage_range)

    selected_ids = select_scene_ids(available, scene_ids)
    selected_records = {scene_id: scene_records[scene_id] for scene_id in selected_ids}

    # Only frame rows for the selected scenes are needed to prepare their caches.
    frame_table = pq.read_table(
        dataset_root / "frames" / f"{split}.parquet", filters=[("scene_id", "in", list(selected_ids))]
    )
    frame_names = index_frame_names(frame_table, selected_records)
    return {
        scene_id: SceneMetadata(scene_record, frame_names[scene_id])
        for scene_id, scene_record in selected_records.items()
    }


def _stage_scene_ids(dataset_root: Path, split: str, stage_range: StageRange) -> tuple[str, ...]:
    """
    Find scenes with at least one query assigned to the requested stages.

    Args:
        dataset_root: Directory containing the stage table.
        split: Validation or test split.
        stage_range: First and last stage to include.

    Returns:
        Sorted scene IDs, with each scene listed once even if it occurs in several stages.
    """
    stage_table = pq.read_table(dataset_root / "stages" / f"{split}.parquet", columns=["scene_id", "stage"])
    scene_ids_by_stage: dict[int, set[str]] = {}
    for stage_row in stage_table.to_pylist():
        scene_ids_by_stage.setdefault(stage_row["stage"], set()).add(stage_row["scene_id"])
    require_stage_range(stage_range, tuple(sorted(scene_ids_by_stage)))
    return tuple(
        sorted(
            {
                scene_id
                for stage in range(stage_range.first, stage_range.last + 1)
                for scene_id in scene_ids_by_stage[stage]
            }
        )
    )
