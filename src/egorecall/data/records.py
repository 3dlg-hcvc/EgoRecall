"""
Typed records for EgoRecall queries and scene annotations. Field names match
the JSON and Parquet files. Object IDs are scoped to scene_id.
"""

import json
from typing import Literal, TypedDict, cast

type QueryKey = tuple[str, int]
type Split = Literal["train", "val", "test"]
type EmissionReason = Literal["new", "answer_change", "rebirth"]
type Program = list[str | int | Program]


class QueryRecord(TypedDict):
    """
    A query, its DSL program, and ground-truth target IDs. query_idx is a stable identifier
    within scene_id; frame is its zero-based canonical query time. The three
    target lists contain object IDs, and program_json stores the nested DSL.
    """

    scene_id: str
    query_idx: int
    split: Split
    description: str
    program_json: str
    source_query_id: str
    program_depth: int
    frame: int
    any_target: bool
    emit_reason: EmissionReason
    target_oids: list[int]
    visible_target_oids: list[int]
    hidden_target_oids: list[int]


class SceneRecord(TypedDict):
    """
    One scene's metadata from scenes.json. Counts cover all rows stored for the
    scene, before stage selection. annotations is a path relative to dataset_root.
    Timeline rates are nominal; exact sensor timestamps are stored in ScanNet++.
    """

    scene_id: str
    split: Split
    num_frames: int
    num_objects: int
    num_queries: int
    source_fps: float
    subsample_factor: int
    nominal_timeline_fps: float
    annotations: str


class FrameVisibility(TypedDict):
    """
    Visible surface-area and image-pixel fractions for one object observation.
    """

    visible_area_frac: float
    visible_pixels_frac: float


class TemporalSummary(TypedDict):
    """
    Object visibility statistics over the complete scene timeline, including
    observations after individual query times. Frame indices are zero-based.
    """

    first_seen_frame: int
    last_seen_frame: int
    peak_visibility_frame: int
    total_visible_frames: int
    peak_visible_area_frac: float


class ObjectAnnotation(TypedDict):
    """
    Full visibility history for an object. Segment endpoints are inclusive;
    per_frame uses canonical frame indices encoded as JSON string keys.
    """

    label: str
    temporal: TemporalSummary
    visibility_segments: list[list[int]]
    per_frame: dict[str, FrameVisibility]


class SceneAnnotations(TypedDict):
    """
    Full-scene visibility annotations for all objects retained by the visibility
    filter, including objects not targeted by selected queries. objects is keyed
    by string object IDs. Image and geometry data are stored separately in ScanNet++.
    """

    schema_version: int
    scene_id: str
    num_frames: int
    visibility_filter: str
    image_pixels: int
    objects: dict[str, ObjectAnnotation]


def decode_program(program_json: str) -> Program:
    """
    Decode program_json into nested DSL arrays. Check array structure and value
    types; operator names and query semantics are not evaluated.

    Args:
        program_json: JSON string whose root and nested subprograms start with
            an operator name, followed by strings, integers, or subprograms.

    Returns:
        The nested program arrays with their ordering and values preserved.
    """
    program = json.loads(program_json)
    if not _is_program(program):
        raise ValueError("program_json must encode an operator-led nested array of strings and integers.")
    return cast(Program, program)


def _is_program(value: object) -> bool:
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
    return all(type(item) in (str, int) or _is_program(item) for item in value[1:])
