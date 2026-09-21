"""
Synthetic datasets with noncontiguous query IDs, shared IDs across scenes,
and several stages for testing data access and selection.
"""

import gzip
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from egorecall.data.schema import FRAME_SCHEMA, QUERY_SCHEMA, STAGE_SCHEMA


@pytest.fixture
def package_root(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    """
    Build a four-query package with deliberately sparse scene-scoped IDs.
    An indirect parameter can select unstaged train instead of staged test.

    Args:
        tmp_path: Isolated directory provided by pytest.
        request: Fixture request with an optional split parameter.

    Returns:
        The synthetic package root.
    """
    split = getattr(request, "param", "test")
    root = tmp_path / "package"
    for directory in ("queries", "stages", "frames", "annotations"):
        (root / directory).mkdir(parents=True)

    # Query IDs are not row positions; ID 4 belongs to two different scenes.
    queries = []
    for scene_id, query_idx, frame in (("scene_a", 4, 0), ("scene_a", 17, 1), ("scene_a", 103, 2), ("scene_b", 4, 1)):
        queries.append(
            {
                "scene_id": scene_id,
                "query_idx": query_idx,
                "split": split,
                "description": "the first chair",
                "program_json": json.dumps(["first_seen", "chair"]),
                "source_query_id": f"source_{query_idx}",
                "program_depth": 1,
                "frame": frame,
                "any_target": False,
                "emit_reason": "new",
                "target_oids": [1],
                "visible_target_oids": [1] if frame < 2 else [],
                "hidden_target_oids": [1] if frame == 2 else [],
            }
        )
    pq.write_table(pa.Table.from_pylist(queries, schema=QUERY_SCHEMA), root / "queries" / f"{split}.parquet")

    # Assign stages independently of query order and preserve a second scene's key.
    assignments = [
        {"scene_id": scene_id, "query_idx": query_idx, "split": split, "stage": stage}
        for scene_id, query_idx, stage in (
            ("scene_b", 4, 2),
            ("scene_a", 103, 3),
            ("scene_a", 4, 1),
            ("scene_a", 17, 2),
        )
    ]
    if split != "train":
        pq.write_table(pa.Table.from_pylist(assignments, schema=STAGE_SCHEMA), root / "stages" / f"{split}.parquet")
    frames = [
        {"scene_id": scene_id, "frame_idx": index, "frame_name": f"frame_{index * 10:06d}"}
        for scene_id in ("scene_a", "scene_b")
        for index in range(3)
    ]
    pq.write_table(pa.Table.from_pylist(frames, schema=FRAME_SCHEMA), root / "frames" / f"{split}.parquet")

    # Retain a contextual object that never appears in these query answers.
    scenes = []
    for scene_id, num_queries in (("scene_a", 3), ("scene_b", 1)):
        scenes.append(
            {
                "scene_id": scene_id,
                "split": split,
                "num_frames": 3,
                "num_objects": 2,
                "num_queries": num_queries,
                "source_fps": 60.0,
                "subsample_factor": 10,
                "nominal_timeline_fps": 6.0,
                "annotations": f"annotations/{scene_id}.json.gz",
            }
        )
        objects = {
            str(oid): {
                "label": label,
                "visibility_segments": [[0, 1]],
                "per_frame": {str(frame): {"visible_area_frac": 0.5, "visible_pixels_frac": 0.1} for frame in (0, 1)},
                "temporal": {
                    "first_seen_frame": 0,
                    "last_seen_frame": 1,
                    "peak_visibility_frame": 0,
                    "total_visible_frames": 2,
                    "peak_visible_area_frac": 0.5,
                },
            }
            for oid, label in ((1, "chair"), (2, "table"))
        }
        annotation = {
            "schema_version": 1,
            "scene_id": scene_id,
            "num_frames": 3,
            "image_pixels": 100,
            "visibility_filter": "visibility_filter_v1",
            "objects": objects,
        }
        (root / "annotations" / f"{scene_id}.json.gz").write_bytes(gzip.compress(json.dumps(annotation).encode()))
    (root / "scenes.json").write_text(json.dumps(scenes))
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "selection": {"split": split, "stage_from": 1, "stage_to": 3},
                "counts": {"queries": 4, "stage_assignments": 0 if split == "train" else 4, "frames": 6, "scenes": 2},
            }
        )
    )
    return root
