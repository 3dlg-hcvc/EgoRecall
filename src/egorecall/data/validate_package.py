"""
Check annotation files before using the readers: required fields, matching IDs,
frame numbering, query answers, visibility histories, and manifest totals.
"""

import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from egorecall.data.integrity import relative_file
from egorecall.data.records import QueryRecord, SceneRecord, decode_program
from egorecall.data.schema import FRAME_SCHEMA, QUERY_SCHEMA, STAGE_SCHEMA
from egorecall.data.validation import (
    is_program,
    require_integer,
    require_text,
    validate_annotations,
    validate_scene,
    validate_table,
)


def validate_frame_mapping(frame_table: pa.Table, scene_records: dict[str, SceneRecord]) -> None:
    """
    Check that each scene has exactly one source filename for every frame number
    from zero to num_frames - 1. Duplicate or missing entries would pair queries
    with the wrong images.

    Args:
        frame_table: Rows from frames/<split>.parquet.
        scene_records: Records from scenes.json keyed by scene ID.
    """
    names_by_index: dict[str, dict[int, str]] = defaultdict(dict)
    for frame_row in frame_table.to_pylist():
        scene_id = frame_row["scene_id"]
        frame_idx = require_integer(frame_row["frame_idx"], "frames/frame_idx")
        frame_name = require_text(frame_row["frame_name"], "frames/frame_name")
        if scene_id not in scene_records:
            raise ValueError(f"frames: unknown scene {scene_id!r}.")
        if frame_idx in names_by_index[scene_id]:
            raise ValueError(f"{scene_id}: duplicate frame index {frame_idx}.")
        names_by_index[scene_id][frame_idx] = frame_name

    for scene_id, scene_record in scene_records.items():
        frame_names = names_by_index[scene_id]
        if set(frame_names) != set(range(scene_record["num_frames"])):
            raise ValueError(f"{scene_id}: incomplete frame mapping.")
        if len(set(frame_names.values())) != len(frame_names):
            raise ValueError(f"{scene_id}: duplicate source frame names.")


def _query_keys(table: pa.Table, split: str, name: str) -> list[tuple[str, int]]:
    """
    Check scene/query IDs before comparing query rows with stage assignments.

    Args:
        table: Query or stage table.
        split: Split that every row must belong to.
        name: Table name for errors.

    Returns:
        Unique (scene_id, query_idx) pairs in file order.
    """
    if any(value != split for value in table["split"].to_pylist()):
        raise ValueError(f"{name}: rows disagree with the requested split {split!r}.")
    query_keys = list(zip(table["scene_id"].to_pylist(), table["query_idx"].to_pylist(), strict=True))
    if len(query_keys) != len(set(query_keys)):
        raise ValueError(f"{name}: duplicate (scene_id, query_idx) keys.")
    for scene_id, query_idx in query_keys:
        require_text(scene_id, f"{name}/scene_id")
        require_integer(query_idx, f"{name}/query_idx")
    return query_keys


def validate_annotation_package(dataset_root: Path) -> None:
    """
    Validate all annotation contents independently of file checksums. This checks
    the full split, including queries outside any later stage selection.

    Args:
        dataset_root: Directory containing manifest.json, tables, and scene annotations.
    """
    with (dataset_root / "manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    version = require_integer(manifest["schema_version"], "manifest/schema_version", minimum=1)
    if version != 1:
        raise ValueError(f"Unsupported package schema_version: {version}.")
    selection = manifest["selection"]
    if not isinstance(selection, dict):
        raise ValueError("manifest/selection must contain a JSON object.")
    split = selection["split"]
    if split not in ("train", "val", "test"):
        raise ValueError(f"Unknown split {split!r}.")
    counts = manifest["counts"]
    if not isinstance(counts, dict):
        raise ValueError("manifest/counts must contain a JSON object.")

    # Scene records define the expected number of frames, queries, and objects.
    with (dataset_root / "scenes.json").open(encoding="utf-8") as stream:
        records = json.load(stream)
    if not isinstance(records, list):
        raise ValueError("scenes.json must contain a list of scene records.")
    scene_records: dict[str, SceneRecord] = {}
    for value in records:
        scene_record = validate_scene(value)
        scene_id = scene_record["scene_id"]
        if scene_id in scene_records:
            raise ValueError(f"Duplicate scene metadata: {scene_id}.")
        if scene_record["split"] != split:
            raise ValueError("Manifest selection does not match the scene split.")
        relative_file(dataset_root, scene_record["annotations"])
        scene_records[scene_id] = scene_record

    query_table = pq.read_table(dataset_root / "queries" / f"{split}.parquet")
    validate_table(query_table, QUERY_SCHEMA, "queries")
    query_keys = _query_keys(query_table, split, "queries")
    query_counts = Counter(scene_id for scene_id, _ in query_keys)
    if query_counts.keys() - scene_records.keys():
        raise ValueError("queries: scene IDs are absent from scenes.json.")
    if not query_keys or any(
        query_counts[scene_id] != record["num_queries"] for scene_id, record in scene_records.items()
    ):
        raise ValueError("queries: row counts disagree with scenes.json or the split is empty.")

    # Every evaluation query needs one stage assignment with the same scene/query ID pair.
    stage_path = dataset_root / "stages" / f"{split}.parquet"
    stage_count = 0
    if split == "train":
        if stage_path.exists():
            raise ValueError("Training must be unstaged; found stages/train.parquet.")
    else:
        stage_table = pq.read_table(stage_path)
        validate_table(stage_table, STAGE_SCHEMA, "stages")
        stage_keys = _query_keys(stage_table, split, "stages")
        if set(stage_keys) != set(query_keys):
            raise ValueError("Query/stage membership differs; every query needs one assignment.")
        stage_values = stage_table["stage"].to_pylist()
        for stage in stage_values:
            require_integer(stage, "stages/stage", minimum=1)
        first = require_integer(selection["stage_from"], "manifest/stage_from", minimum=1)
        last = require_integer(selection["stage_to"], "manifest/stage_to", minimum=first)
        available = sorted(set(stage_values))
        if len(available) != last - first + 1 or available[0] != first or available[-1] != last:
            raise ValueError("Packaged stages disagree with the range declared in manifest.json.")
        stage_count = len(stage_keys)

    frame_table = pq.read_table(dataset_root / "frames" / f"{split}.parquet")
    validate_table(frame_table, FRAME_SCHEMA, "frames")
    validate_frame_mapping(frame_table, scene_records)

    # Collect answer IDs so visibility files can be checked for every target, not just queried scenes.
    target_ids: dict[str, set[int]] = defaultdict(set)
    any_target_count = 0
    for batch in query_table.to_batches(max_chunksize=8192):
        for query_record in batch.to_pylist():
            validate_query(query_record, scene_records[query_record["scene_id"]])
            target_ids[query_record["scene_id"]].update(query_record["target_oids"])
            any_target_count += query_record["any_target"]

    object_count = 0
    for scene_id, scene_record in scene_records.items():
        with gzip.open(relative_file(dataset_root, scene_record["annotations"]), "rt", encoding="utf-8") as stream:
            scene_annotations = validate_annotations(json.load(stream), scene_record)
        object_ids = {int(oid) for oid in scene_annotations["objects"]}
        missing = target_ids[scene_id] - object_ids
        if missing:
            raise ValueError(f"{scene_id}: target IDs have no annotation: {sorted(missing)}.")
        object_count += len(object_ids)

    observed = dict(
        queries=len(query_keys),
        stage_assignments=stage_count,
        frames=frame_table.num_rows,
        scenes=len(scene_records),
        objects=object_count,
        any_target_queries=any_target_count,
    )
    for name, actual in observed.items():
        declared = require_integer(counts[name], f"manifest/counts/{name}")
        if declared != actual:
            raise ValueError(f"manifest/counts/{name}: declared {declared}, found {actual}.")


def validate_query(query: QueryRecord, scene_record: SceneRecord) -> None:
    """
    Check query time against the scene's frame count, validate the target-ID
    partition, and check the nested program representation.

    Args:
        query: Query row to check.
        scene_record: Counts and settings for the scene containing the query.
    """
    scene_id, query_idx = query["scene_id"], query["query_idx"]
    context = f"{scene_id}/{query_idx}"
    frame = require_integer(query["frame"], f"{context}/frame")
    if frame >= scene_record["num_frames"]:
        raise ValueError(f"{context}: query frame is outside the scene frame range.")

    require_text(query["description"], f"{context}/description")
    require_text(query["source_query_id"], f"{context}/source_query_id")

    reason = query["emit_reason"]
    if reason not in ("new", "answer_change", "rebirth"):
        raise ValueError(f"{context}: unknown emit_reason {reason!r}.")

    # Every target occurs once, and visible/hidden IDs partition the answer.
    for field in ("target_oids", "visible_target_oids", "hidden_target_oids"):
        ids = query[field]
        for oid in ids:
            require_integer(oid, f"{context}/{field}", minimum=1)
        if len(ids) != len(set(ids)):
            raise ValueError(f"{context}: duplicate object IDs in {field}.")

    targets = set(query["target_oids"])
    visible, hidden = set(query["visible_target_oids"]), set(query["hidden_target_oids"])
    if not targets or targets != visible | hidden or visible & hidden:
        raise ValueError(f"{context}: visible/hidden IDs do not partition the target IDs.")

    # Validate program metadata and its nested representation together.
    require_integer(query["program_depth"], f"{context}/program_depth", minimum=1)
    try:
        program = decode_program(query["program_json"])
        if not is_program(program):
            raise ValueError("Invalid program structure.")
    except ValueError as error:
        raise ValueError(f"{context}: invalid program_json: {error}") from error
