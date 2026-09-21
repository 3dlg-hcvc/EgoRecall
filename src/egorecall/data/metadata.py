"""
Load scene metadata and frame mappings without reading query contents or visibility annotations.
"""

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from egorecall.data.records import SceneRecord
from egorecall.data.schema import FRAME_SCHEMA, STAGE_SCHEMA, validate_table
from egorecall.data.stages import StageRange, parse_stages, require_stage_range
from egorecall.data.validation import require_integer, require_text, validate_scene


@dataclass(frozen=True)
class SceneMetadata:
    """
    Scene metadata and the complete canonical frame mapping needed for preparation.

    Args:
        metadata: Scene record containing sampling settings and population counts.
        frame_names: Source names ordered by canonical frame index.
    """

    metadata: SceneRecord
    frame_names: tuple[str, ...]


def select_scene_ids(available: tuple[str, ...], requested: list[str] | None) -> tuple[str, ...]:
    """
    Restrict scene work to a nonempty, unique subset of an available selection.

    Args:
        available: Scene IDs represented in the selected split/stages.
        requested: Explicit IDs to retain, or None for the complete available selection.

    Returns:
        Scene IDs in the supplied order, or the available order when omitted.
    """
    if requested is None:
        return available

    if not requested or len(requested) != len(set(requested)):
        raise ValueError("Scene selection must be nonempty and contain no duplicates.")

    missing = set(requested) - set(available)
    if missing:
        raise KeyError(f"Scenes are absent from the requested split/stage selection: {sorted(missing)}.")
    return tuple(requested)


def index_frame_names(table: pa.Table, scenes: dict[str, SceneRecord]) -> dict[str, tuple[str, ...]]:
    """
    Validate a scene/frame join and order names by their explicit canonical index.

    Args:
        table: Schema-validated frame rows for the scenes to index.
        scenes: Metadata defining each scene's frame count.

    Returns:
        Complete canonical frame-name sequences keyed by scene.
    """
    by_scene: dict[str, dict[int, str]] = defaultdict(dict)
    for batch in table.to_batches(max_chunksize=8192):
        for row in batch.to_pylist():
            scene_id, frame_idx, name = row["scene_id"], row["frame_idx"], row["frame_name"]
            if scene_id not in scenes:
                raise ValueError(f"frames: unknown scene {scene_id!r} in the scene metadata.")
            require_integer(frame_idx, "frames/frame_idx")
            require_text(name, "frames/frame_name")
            if frame_idx in by_scene[scene_id]:
                raise ValueError(f"{scene_id}: duplicate canonical frame index {frame_idx}.")
            by_scene[scene_id][frame_idx] = name

    ordered: dict[str, tuple[str, ...]] = {}
    for scene_id, scene in scenes.items():
        frames = by_scene[scene_id]
        if set(frames) != set(range(scene["num_frames"])) or len(set(frames.values())) != len(frames):
            raise ValueError(f"{scene_id}: incomplete or ambiguous canonical frame mapping.")
        ordered[scene_id] = tuple(frames[i] for i in range(scene["num_frames"]))
    return ordered


def load_scene_metadata(
    dataset_root: Path,
    split: str,
    *,
    stages: int | str | None = None,
    scene_ids: list[str] | None = None,
) -> dict[str, SceneMetadata]:
    """
    Read scene metadata and selected frame mappings for preparation. Read stage
    assignments only when a stage selection is supplied. Query Parquet files and
    per-scene visibility payloads are not needed for this operation.

    Check the package format, scene/stage selection, and selected frame mappings.
    Use the dataset checker for package-wide counts and query membership checks.

    Args:
        dataset_root: EgoRecall directory containing manifest.json and scenes.json.
        split: Requested split, matching the single-split package manifest.
        stages: Exact stage, inclusive range, or None for all represented scenes.
        scene_ids: Optional scene subset, preserving the supplied order.

    Returns:
        Scene records and complete frame mappings for the selected scenes.
    """
    root = dataset_root.expanduser().resolve()

    if split not in ("train", "val", "test"):
        raise ValueError(f"Unknown split {split!r}; use train, val, or test.")

    stage_range = parse_stages(stages) if stages is not None else None
    if split == "train" and stage_range is not None:
        raise ValueError("Training is unstaged; omit stages when reading train.")

    # Identify the package format and split before reading scene records.
    with (root / "manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)

    version = require_integer(manifest["schema_version"], "manifest/schema_version", minimum=1)
    if version != 1:
        raise ValueError(f"Unsupported package schema_version: {version}.")

    if manifest["selection"]["split"] != split:
        raise ValueError(f"Manifest selection does not match the requested {split} split.")

    # Find scenes represented in the requested stages, then apply any explicit scene subset.
    scenes = _read_scenes(root, split)
    available = tuple(sorted(scene_id for scene_id, scene in scenes.items() if scene["num_queries"] > 0))
    if stage_range is not None:
        available = _stage_scenes(root, split, stage_range)

    selected = select_scene_ids(available, scene_ids)
    selected_scenes = {scene_id: scenes[scene_id] for scene_id in selected}

    # Read and index complete frame mappings for the selected scenes only.
    frames = pq.read_table(root / "frames" / f"{split}.parquet", filters=[("scene_id", "in", list(selected))])
    validate_table(frames, FRAME_SCHEMA, "frames")
    names = index_frame_names(frames, selected_scenes)
    return {scene_id: SceneMetadata(scene, names[scene_id]) for scene_id, scene in selected_scenes.items()}


def _read_scenes(root: Path, split: str) -> dict[str, SceneRecord]:
    """
    Read scene records containing sampling settings and frame counts.

    Args:
        root: Annotation package directory.
        split: Requested benchmark split.

    Returns:
        Scene records for the split, keyed by unique scene ID.
    """
    with (root / "scenes.json").open(encoding="utf-8") as stream:
        records = json.load(stream)

    # Validate records and their identities before selecting the split.
    scenes: dict[str, SceneRecord] = {}
    for value in records:
        scene = validate_scene(value)
        scene_id = scene["scene_id"]
        if scene_id in scenes:
            raise ValueError(f"Duplicate scene metadata: {scene_id}.")
        scenes[scene_id] = scene

    scenes = {scene_id: scene for scene_id, scene in scenes.items() if scene["split"] == split}
    if not scenes:
        raise ValueError(f"Split {split!r} is not packaged.")
    return scenes


def _stage_scenes(root: Path, split: str, requested: StageRange) -> tuple[str, ...]:
    """
    Select scenes from stage assignments using only scene, split, and stage columns.

    Args:
        root: Annotation package directory.
        split: Requested validation or test split.
        requested: Stage bounds to select.

    Returns:
        Sorted scene IDs with assignments in the requested range.
    """
    # Read the columns needed to select scenes and check the requested stage range.
    columns = ["scene_id", "split", "stage"]
    table = pq.read_table(root / "stages" / f"{split}.parquet", columns=columns)
    validate_table(table, pa.schema([STAGE_SCHEMA.field(name) for name in columns]), "stages")
    if pc.unique(table["split"]).to_pylist() != [split]:
        raise ValueError(f"stages: rows disagree with the requested split {split!r}.")

    available = tuple(sorted(pc.unique(table["stage"]).to_pylist()))
    require_stage_range(requested, available)

    # Multiple query assignments can refer to the same scene.
    mask = pc.and_(pc.greater_equal(table["stage"], requested.first), pc.less_equal(table["stage"], requested.last))
    return tuple(sorted(pc.unique(pc.filter(table["scene_id"], mask)).to_pylist()))
