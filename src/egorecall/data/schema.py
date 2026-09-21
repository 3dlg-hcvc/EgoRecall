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


def validate_table(table: pa.Table, schema: pa.Schema, name: str) -> None:
    """
    Require the expected Arrow columns/types and reject null fields at the read boundary.

    Args:
        table: Loaded table or column projection.
        schema: Expected schema for those columns.
        name: Table description used in validation errors.
    """
    if not table.schema.equals(schema, check_metadata=False):
        raise ValueError(f"{name}: incompatible schema; expected {schema.names}, got {table.column_names}.")
    if any(column.null_count for column in table.columns):
        raise ValueError(f"{name}: null table fields are not allowed.")
