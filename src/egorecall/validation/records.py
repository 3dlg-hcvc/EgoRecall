"""
Validate one record at a time: JSON field types, required keys, and scene IDs, object IDs, and
frame indices in scene records, query rows, and scene annotations, plus table schemas.
"""

from typing import cast

import pyarrow as pa

from egorecall.arguments import require_integer, require_number, require_text
from egorecall.data.records import (
    FrameVisibility,
    ObjectAnnotation,
    QueryRecord,
    SceneAnnotations,
    SceneRecord,
    TemporalSummary,
    decode_program,
)


def require_fields(value: object, fields: frozenset[str], context: str) -> dict[str, object]:
    """
    Require a dictionary with exactly the expected field names.

    Args:
        value: Decoded JSON value.
        fields: Expected field names.
        context: Record description used in validation errors.

    Returns:
        The input dictionary after checking its field names.
    """
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{context}: expected fields {sorted(fields)}.")
    return cast(dict[str, object], value)


def validate_scene(value: object) -> SceneRecord:
    """
    Check sampling settings, counts, and the visibility filename in one scenes.json
    record. Counts describe all data stored for the scene, before selecting stages.

    Args:
        value: One decoded entry from scenes.json.

    Returns:
        The input scene record with validated field types.
    """
    record = require_fields(value, SceneRecord.__required_keys__, "Scene metadata")
    scene_id = require_text(record["scene_id"], "scene_id")

    split = record["split"]
    if split not in ("train", "val", "test"):
        raise ValueError(f"{scene_id}: unknown split {split!r}.")

    # Check the count and frame-rate fields in the scene metadata.
    for field in ("num_frames", "subsample_factor"):
        require_integer(record[field], f"{scene_id}/{field}", minimum=1)
    for field in ("num_queries", "num_objects"):
        require_integer(record[field], f"{scene_id}/{field}")

    for field in ("source_fps", "nominal_timeline_fps"):
        require_number(record[field], f"{scene_id}/{field}")

    require_text(record["annotations"], f"{scene_id}/annotations")
    return cast(SceneRecord, record)


def validate_annotations(value: object, scene_record: SceneRecord) -> SceneAnnotations:
    """
    Validate a scene's annotation structure, object IDs, and frame indices.
    Histories cover the complete scene timeline, including observations after
    individual query times.

    Args:
        value: Decoded per-scene JSON annotation.
        scene_record: Expected scene ID, frame count, and object count from scenes.json.

    Returns:
        The input annotation record with validated nested field types.
    """
    scene_id = scene_record["scene_id"]
    record = require_fields(value, SceneAnnotations.__required_keys__, f"{scene_id}/annotations")
    if type(record["schema_version"]) is not int or record["schema_version"] != 1:
        raise ValueError(f"{scene_id}: unsupported annotation schema_version.")

    num_frames = require_integer(record["num_frames"], f"{scene_id}/num_frames", minimum=1)
    if record["scene_id"] != scene_id or num_frames != scene_record["num_frames"]:
        raise ValueError(f"{scene_id}: annotation scene or timeline does not match scenes.json.")

    require_text(record["visibility_filter"], f"{scene_id}/visibility_filter")
    require_integer(record["image_pixels"], f"{scene_id}/image_pixels", minimum=1)

    # Validate every retained object, including contextual objects not in answers.
    object_annotations = record["objects"]
    if not isinstance(object_annotations, dict) or len(object_annotations) != scene_record["num_objects"]:
        raise ValueError(f"{scene_id}: annotation object count does not match scenes.json.")
    for oid, object_annotation in object_annotations.items():
        if not isinstance(oid, str) or not oid.isdecimal() or int(oid) <= 0 or str(int(oid)) != oid:
            raise ValueError(f"{scene_id}: invalid annotation object ID {oid!r}.")
        _validate_object(object_annotation, num_frames, f"{scene_id}/{oid}")

    return cast(SceneAnnotations, record)


def _validate_object(value: object, num_frames: int, context: str) -> None:
    """
    Check object field types and visibility indices against the scene timeline.

    Args:
        value: One decoded object annotation.
        num_frames: Number of sampled frames in the scene.
        context: Scene/object description used in errors.
    """
    object_annotation = require_fields(value, ObjectAnnotation.__required_keys__, context)
    require_text(object_annotation["label"], f"{context}/label")

    # Check timeline bounds and aggregate visibility statistics.
    temporal = require_fields(object_annotation["temporal"], TemporalSummary.__required_keys__, f"{context}/temporal")
    for field in ("first_seen_frame", "last_seen_frame", "peak_visibility_frame"):
        frame = require_integer(temporal[field], f"{context}/{field}")
        if frame >= num_frames:
            raise ValueError(f"{context}/{field}: frame outside the sampled frame sequence.")

    count = require_integer(temporal["total_visible_frames"], f"{context}/total_visible_frames", minimum=1)
    if count > num_frames:
        raise ValueError(f"{context}: total_visible_frames exceeds the scene length.")
    require_number(temporal["peak_visible_area_frac"], f"{context}/peak_visible_area_frac")

    # Segments and observations both use sampled frame indices; segment ends are inclusive.
    segments = object_annotation["visibility_segments"]
    if not isinstance(segments, list):
        raise ValueError(f"{context}: visibility_segments must be a list.")
    for segment in segments:
        if not isinstance(segment, list) or len(segment) != 2:
            raise ValueError(f"{context}: each visibility segment must contain two endpoints.")
        start = require_integer(segment[0], f"{context}/segment start")
        end = require_integer(segment[1], f"{context}/segment end")
        if not start <= end < num_frames:
            raise ValueError(f"{context}: invalid visibility segment bounds.")

    observations = object_annotation["per_frame"]
    if not isinstance(observations, dict):
        raise ValueError(f"{context}: per_frame must be a dictionary.")
    for frame_key, statistics in observations.items():
        if not isinstance(frame_key, str) or not frame_key.isdecimal():
            raise ValueError(f"{context}: invalid observation frame {frame_key!r}.")
        frame_idx = int(frame_key)
        if str(frame_idx) != frame_key or frame_idx >= num_frames:
            raise ValueError(f"{context}: observation frame outside the sampled frame sequence.")

        values = require_fields(statistics, FrameVisibility.__required_keys__, f"{context}/{frame_key}")
        for name, fraction in values.items():
            require_number(fraction, f"{context}/{frame_key}/{name}")


def validate_table(table: pa.Table, schema: pa.Schema, name: str) -> None:
    """
    Require the expected Arrow columns and types, and reject null values.

    Args:
        table: Table read from a Parquet file.
        schema: Expected column names and types.
        name: Table description used in validation errors.
    """
    if not table.schema.equals(schema, check_metadata=False):
        raise ValueError(f"{name}: incompatible schema; expected {schema.names}, got {table.column_names}.")
    if any(column.null_count for column in table.columns):
        raise ValueError(f"{name}: null table fields are not allowed.")


def is_program(value: object) -> bool:
    """
    Check nested program structure, treating booleans as invalid integer
    operands even though bool subclasses int in Python.

    Args:
        value: A decoded JSON value to inspect.

    Returns:
        Whether the value is an operator-led array of valid operands or subprograms.
    """
    if not isinstance(value, list) or not value or not isinstance(value[0], str) or not value[0]:
        return False
    return all(type(item) in (str, int) or is_program(item) for item in value[1:])


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
