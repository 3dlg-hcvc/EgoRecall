"""
Column names and Arrow types for query, stage, and frame tables. The reader validates
names and types before joining records or interpreting integer identifiers.
"""

import pyarrow as pa

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
