"""
Typed records and Arrow table schemas for EgoRecall queries, stages, frames, scene
annotations, and the manifest. Field names match the JSON and Parquet files. Object
IDs are scoped to scene_id.
"""

import json
from typing import Literal, TypedDict, cast

import pyarrow as pa

type QueryKey = tuple[str, int]
type Split = Literal["train", "val", "test"]
type EmissionReason = Literal["new", "answer_change", "rebirth"]
type Program = list[str | int | Program]

# Column names and Arrow types for the query, stage, and frame tables. The standalone
# checker compares these schemas with the Parquet tables before the readers use them.
OBJECT_IDS = pa.list_(pa.int32())
QUERY_SCHEMA = pa.schema(
    [
        ("scene_id", pa.string()),
        ("query_idx", pa.int32()),
        ("split", pa.string()),
        ("description", pa.string()),
        ("program_json", pa.string()),
        ("source_query_id", pa.string()),
        ("program_depth", pa.int32()),
        ("frame", pa.int32()),
        ("any_target", pa.bool_()),
        ("emit_reason", pa.string()),
        ("target_oids", OBJECT_IDS),
        ("visible_target_oids", OBJECT_IDS),
        ("hidden_target_oids", OBJECT_IDS),
    ]
)

STAGE_SCHEMA = pa.schema(
    [
        ("scene_id", pa.string()),
        ("query_idx", pa.int32()),
        ("split", pa.string()),
        ("stage", pa.int32()),
    ]
)

FRAME_SCHEMA = pa.schema(
    [
        ("scene_id", pa.string()),
        ("frame_idx", pa.int32()),
        ("frame_name", pa.string()),
    ]
)


class QueryRecord(TypedDict):
    """
    A query, its DSL program, and ground-truth target IDs. query_idx is a stable identifier
    within scene_id; frame is its zero-based sampled frame number. The three
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
    per_frame uses sampled frame indices encoded as JSON string keys.
    """

    label: str
    temporal: TemporalSummary
    visibility_segments: list[list[int]]
    per_frame: dict[str, FrameVisibility]


class SceneAnnotations(TypedDict):
    """
    Full-scene visibility annotations for all objects retained by the visibility
    filter, including objects not targeted by selected queries. objects is keyed
    by string object IDs. Images and object boxes are stored separately in the prepared H5 cache.
    """

    schema_version: int
    scene_id: str
    num_frames: int
    visibility_filter: str
    image_pixels: int
    objects: dict[str, ObjectAnnotation]


class StageBounds(TypedDict):
    """
    Inclusive stage range for a validation or test split.
    """

    first: int
    last: int


class SplitManifest(TypedDict):
    """
    Counts for one split: queries, stage assignments, frames, scenes, objects, and queries with any_target set.
    stages gives its inclusive first/last stage range, or None for training.
    """

    counts: dict[str, int]
    stages: StageBounds | None


def decode_program(program_json: str) -> Program:
    """
    Decode program_json into nested lists of operator names and arguments.

    Args:
        program_json: JSON string whose root and nested subprograms start with
            an operator name, followed by strings, integers, or subprograms.

    Returns:
        The nested program arrays with their ordering and values preserved.
    """
    program = json.loads(program_json)
    return cast(Program, program)
