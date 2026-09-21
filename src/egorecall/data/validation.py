"""
Validate JSON field types, required keys, and scene/object/frame identities
before returning typed annotation records.
"""

import math
from typing import cast

from egorecall.data.records import FrameVisibility, ObjectAnnotation, SceneAnnotations, SceneRecord, TemporalSummary


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


def require_integer(value: object, context: str, minimum: int = 0) -> int:
    """
    Require a Python integer at or above minimum. Booleans are rejected.

    Args:
        value: Value to check.
        context: Field description used in errors.
        minimum: Smallest accepted value.

    Returns:
        The validated integer.
    """
    if type(value) is not int or value < minimum:
        raise ValueError(f"{context}: expected an integer >= {minimum}.")
    return value


def require_text(value: object, context: str) -> str:
    """
    Require a string containing at least one non-whitespace character.

    Args:
        value: Value to check.
        context: Field description used in errors.

    Returns:
        The input string, including any surrounding whitespace.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: expected a nonempty string.")
    return value


def require_number(value: object, context: str) -> None:
    """
    Require a finite, nonnegative numeric statistic. Validation leaves its value unchanged.

    Args:
        value: Numeric statistic to check.
        context: Field description used in errors.
    """
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{context}: expected a finite, nonnegative number.")


def validate_scene(value: object) -> SceneRecord:
    """
    Check the scene metadata fields needed for split, frame, and annotation
    joins. Counts describe the package before a reader's stage selection.

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


def validate_annotations(value: object, scene: SceneRecord) -> SceneAnnotations:
    """
    Validate a scene's annotation structure and object/frame identities.
    Histories cover the complete scene timeline, including observations after
    individual query times.

    Args:
        value: Decoded per-scene JSON annotation.
        scene: Metadata to join against.

    Returns:
        The input annotation record with validated nested field types.
    """
    scene_id = scene["scene_id"]
    record = require_fields(value, SceneAnnotations.__required_keys__, f"{scene_id}/annotations")
    if type(record["schema_version"]) is not int or record["schema_version"] != 1:
        raise ValueError(f"{scene_id}: unsupported annotation schema_version.")
    n_frames = require_integer(record["num_frames"], f"{scene_id}/num_frames", minimum=1)
    if record["scene_id"] != scene_id or n_frames != scene["num_frames"]:
        raise ValueError(f"{scene_id}: annotation scene or timeline does not match scenes.json.")
    require_text(record["visibility_filter"], f"{scene_id}/visibility_filter")
    require_integer(record["image_pixels"], f"{scene_id}/image_pixels", minimum=1)

    # Validate every retained object, including contextual objects not in answers.
    objects = record["objects"]
    if not isinstance(objects, dict) or len(objects) != scene["num_objects"]:
        raise ValueError(f"{scene_id}: annotation object count does not match scenes.json.")
    for oid, obj in objects.items():
        if not isinstance(oid, str) or not oid.isdecimal() or int(oid) <= 0 or str(int(oid)) != oid:
            raise ValueError(f"{scene_id}: invalid annotation object ID {oid!r}.")
        _validate_object(obj, n_frames, f"{scene_id}/{oid}")

    return cast(SceneAnnotations, record)


def _validate_object(value: object, n_frames: int, context: str) -> None:
    """
    Check object field types and visibility indices against the scene timeline.

    Args:
        value: One decoded object annotation.
        n_frames: Number of canonical frames in the scene.
        context: Scene/object description used in errors.
    """
    obj = require_fields(value, ObjectAnnotation.__required_keys__, context)
    require_text(obj["label"], f"{context}/label")
    temporal = require_fields(obj["temporal"], TemporalSummary.__required_keys__, f"{context}/temporal")
    for field in ("first_seen_frame", "last_seen_frame", "peak_visibility_frame"):
        frame = require_integer(temporal[field], f"{context}/{field}")
        if frame >= n_frames:
            raise ValueError(f"{context}/{field}: frame outside the canonical timeline.")
    count = require_integer(temporal["total_visible_frames"], f"{context}/total_visible_frames", minimum=1)
    if count > n_frames:
        raise ValueError(f"{context}: total_visible_frames exceeds the scene length.")
    require_number(temporal["peak_visible_area_frac"], f"{context}/peak_visible_area_frac")

    # Segments and observations both use canonical indices; segment ends are inclusive.
    segments = obj["visibility_segments"]
    if not isinstance(segments, list):
        raise ValueError(f"{context}: visibility_segments must be a list.")
    for segment in segments:
        if not isinstance(segment, list) or len(segment) != 2:
            raise ValueError(f"{context}: each visibility segment must contain two endpoints.")
        start = require_integer(segment[0], f"{context}/segment start")
        end = require_integer(segment[1], f"{context}/segment end")
        if not start <= end < n_frames:
            raise ValueError(f"{context}: invalid visibility segment bounds.")

    observations = obj["per_frame"]
    if not isinstance(observations, dict):
        raise ValueError(f"{context}: per_frame must be a dictionary.")
    for frame_key, statistics in observations.items():
        if not isinstance(frame_key, str) or not frame_key.isdecimal():
            raise ValueError(f"{context}: invalid observation frame {frame_key!r}.")
        frame_idx = int(frame_key)
        if str(frame_idx) != frame_key or frame_idx >= n_frames:
            raise ValueError(f"{context}: observation frame outside the canonical timeline.")
        values = require_fields(statistics, FrameVisibility.__required_keys__, f"{context}/{frame_key}")
        for name, fraction in values.items():
            require_number(fraction, f"{context}/{frame_key}/{name}")
