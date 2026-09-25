"""
Check the annotation package as a whole before using the readers: file checksums and
splits declared in manifest.json, matching IDs across the query, stage, and frame tables,
visibility files, and manifest totals. Checks of single records are in validation/records.py.
"""

import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq

from egorecall.arguments import require_integer, require_text
from egorecall.data.records import FRAME_SCHEMA, QUERY_SCHEMA, STAGE_SCHEMA, SceneRecord, SplitManifest
from egorecall.data.scannetpp import source_frame_index
from egorecall.integrity import fingerprint_file, relative_file
from egorecall.validation.records import validate_annotations, validate_query, validate_scene, validate_table


def verify_package(root: Path) -> int:
    """
    Verify manifest byte counts and SHA-256 hashes, requiring coverage of all data
    files used by the reader. File symlinks used by download caches are supported.

    Args:
        root: EgoRecall dataset directory.

    Returns:
        Number of files verified against the manifest.
    """
    with (root / "manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    files = manifest["files"]
    if not isinstance(files, dict) or not files:
        raise ValueError("manifest/files must be a nonempty mapping of filenames to fingerprints.")

    for name, expected in files.items():
        path = relative_file(root, name)
        require_integer(expected["bytes"], f"manifest/files/{name}/bytes")
        actual = fingerprint_file(path)
        if actual != expected:
            raise ValueError(f"{name}: bytes or SHA-256 do not match manifest.json.")

    # A valid checksum list must cover the tables and scene files used for loading.
    required = {"scenes.json"}
    for split in manifest_splits(manifest):
        required.update((f"queries/{split}.parquet", f"frames/{split}.parquet"))
        if split != "train":
            required.add(f"stages/{split}.parquet")

    with (root / "scenes.json").open(encoding="utf-8") as stream:
        required.update(scene_record["annotations"] for scene_record in json.load(stream))
    missing = required - files.keys()
    if missing:
        raise ValueError(f"Manifest omits required files: {sorted(missing)}.")
    return len(files)


def manifest_splits(manifest: dict[str, object]) -> dict[str, SplitManifest]:
    """
    Check the splits mapping in manifest.json, then return each split's counts and stage range.
    The manifest lists each split under splits and the whole dataset's totals under counts.

    Args:
        manifest: JSON object read from manifest.json.

    Returns:
        Records keyed by train, val, or test, each containing counts and an inclusive
        first/last stage range. Training records have stages set to None.
    """
    version = require_integer(manifest["schema_version"], "manifest/schema_version", minimum=1)
    if version != 2:
        raise ValueError(f"Unsupported package schema_version: {version}.")
    if not isinstance(manifest["counts"], dict):
        raise ValueError("manifest/counts must contain a JSON object.")
    splits = manifest["splits"]
    if not isinstance(splits, dict) or not splits:
        raise ValueError("manifest/splits must contain a nonempty split mapping.")

    # Training has no stage assignments; validation and test need an inclusive stage range.
    for split, split_manifest in splits.items():
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split {split!r}.")
        if not isinstance(split_manifest["counts"], dict):
            raise ValueError(f"manifest/splits/{split}/counts must contain a JSON object.")
        stages = split_manifest["stages"]
        if split == "train":
            if stages is not None:
                raise ValueError("Training must be unstaged.")
        else:
            if not isinstance(stages, dict):
                raise ValueError(f"{split}: manifest stages must contain first/last bounds.")
            first = require_integer(stages["first"], f"manifest/splits/{split}/stages/first", minimum=1)
            require_integer(stages["last"], f"manifest/splits/{split}/stages/last", minimum=first)
    return cast(dict[str, SplitManifest], splits)


def validate_frame_mapping(frame_table: pa.Table, scene_records: dict[str, SceneRecord]) -> None:
    """
    Check that each scene has exactly one ScanNet++ frame name for every frame number
    from zero to num_frames - 1, with source indices increasing in frame-number order.
    Duplicate, missing, or reordered entries would pair queries with the wrong images.

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

        # Preparation extracts source frames in video order, so the names must be ScanNet++ frame
        # names whose source indices increase with frame_idx; this also rules out duplicate names.
        source_indices = [source_frame_index(frame_names[frame_idx]) for frame_idx in range(scene_record["num_frames"])]
        if any(earlier >= later for earlier, later in zip(source_indices, source_indices[1:])):
            raise ValueError(f"{scene_id}: frame names must increase with frame_idx.")


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


def validate_annotation_package(dataset_root: Path) -> dict[str, dict[str, int]]:
    """
    Check all query, stage, frame, and visibility records against the scene metadata
    and manifest counts. This covers every stored query, regardless of the stages a
    reader selects. File checksums are checked separately by verify_package().

    Args:
        dataset_root: Directory containing manifest.json, tables, and scene annotations.

    Returns:
        Observed counts for each checked split.
    """
    with (dataset_root / "manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    split_manifests = manifest_splits(manifest)

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
        if scene_record["split"] not in split_manifests:
            raise ValueError(f"{scene_id}: split {scene_record['split']!r} is not listed in manifest.json.")
        relative_file(dataset_root, scene_record["annotations"])
        scene_records[scene_id] = scene_record

    # Check each split independently before comparing their combined counts with the manifest.
    split_counts = {}
    for split, split_manifest in split_manifests.items():
        selected_records = {
            scene_id: scene_record for scene_id, scene_record in scene_records.items() if scene_record["split"] == split
        }
        split_counts[split] = _validate_split(dataset_root, split, selected_records, split_manifest)

    for name in ("queries", "stage_assignments", "frames", "scenes", "objects", "any_target_queries"):
        observed = sum(counts[name] for counts in split_counts.values())
        declared = require_integer(manifest["counts"][name], f"manifest/counts/{name}")
        if declared != observed:
            raise ValueError(f"manifest/counts/{name}: declared {declared}, found {observed}.")
    return split_counts


def _validate_split(
    dataset_root: Path, split: str, scene_records: dict[str, SceneRecord], split_manifest: SplitManifest
) -> dict[str, int]:
    """
    Check one split's IDs, answers, stage assignments, frame names, and visibility files.

    Args:
        dataset_root: Annotation package directory.
        split: Split whose tables are checked.
        scene_records: Scene metadata belonging to this split.
        split_manifest: Expected counts and stage bounds.

    Returns:
        Counts computed from this split's actual contents.
    """
    counts = split_manifest["counts"]
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
        first = require_integer(split_manifest["stages"]["first"], f"manifest/splits/{split}/stages/first", minimum=1)
        last = require_integer(split_manifest["stages"]["last"], f"manifest/splits/{split}/stages/last", minimum=first)
        available = sorted(set(stage_values))
        if len(available) != last - first + 1 or available[0] != first or available[-1] != last:
            raise ValueError("Packaged stages disagree with the range declared in manifest.json.")
        stage_count = len(stage_keys)

    frame_table = pq.read_table(dataset_root / "frames" / f"{split}.parquet")
    validate_table(frame_table, FRAME_SCHEMA, "frames")
    validate_frame_mapping(frame_table, scene_records)

    # Collect each scene's answer IDs to check that every target has a visibility record.
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
        declared = require_integer(counts[name], f"manifest/splits/{split}/counts/{name}")
        if declared != actual:
            raise ValueError(f"manifest/splits/{split}/counts/{name}: declared {declared}, found {actual}.")

    return observed
