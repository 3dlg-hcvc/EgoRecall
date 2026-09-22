"""
Read queries and visibility annotations after checking the dataset with egorecall-check.
"""

import gzip
import json
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from egorecall.arguments import require_integer
from egorecall.data.metadata import index_frame_names, read_scene_records
from egorecall.data.records import QueryKey, QueryRecord, SceneAnnotations, SceneRecord, Split
from egorecall.data.stages import parse_stages, require_stage_range


class EgoRecallAnnotations:
    """
    Read a split's query table and select queries by their stage assignments.
    Visibility files are loaded separately by get_annotations(). Run egorecall-check
    before using a new or changed dataset; loading does not repeat those checks.

    Args:
        dataset_root: Directory containing scenes.json and the Parquet tables.
        split: One of train, val, or test.
        stages: Exact stage, inclusive LO:HI range, or None for all queries. Omit for training.
    """

    def __init__(self, dataset_root: Path, split: str = "test", stages: int | str | None = None) -> None:
        """
        Load query rows and build lookups by (scene_id, query_idx). Query IDs can
        have gaps, so they cannot be used directly as table row numbers.

        Args:
            dataset_root: Directory containing the annotation data.
            split: Query split to read.
            stages: Optional evaluation stage selection.
        """
        if not isinstance(dataset_root, Path):
            raise TypeError("dataset_root must be a pathlib.Path.")
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split {split!r}; use train, val, or test.")

        self.root = dataset_root.expanduser().resolve()
        self.split = cast(Split, split)
        self.stage_range = parse_stages(stages) if stages is not None else None
        if self.split == "train":
            if self.stage_range is not None:
                raise ValueError("Training is unstaged; omit stages when reading train.")

        scene_records = read_scene_records(self.root)
        represented_splits = {scene_record["split"] for scene_record in scene_records.values()}
        self.available_splits = tuple(name for name in ("train", "val", "test") if name in represented_splits)

        # Select stage assignments first so a small evaluation subset does not load every query.
        stage_by_key: dict[QueryKey, int] = {}
        query_filters = None
        self.available_stages: tuple[int, ...] = ()
        if split != "train":
            stage_table = pq.read_table(self.root / "stages" / f"{split}.parquet")
            self.available_stages = tuple(sorted(pc.unique(stage_table["stage"]).to_pylist()))
            
            if self.stage_range is not None:
                require_stage_range(self.stage_range, self.available_stages)
                stage_table = stage_table.filter(
                    pc.and_(
                        pc.greater_equal(stage_table["stage"], self.stage_range.first),
                        pc.less_equal(stage_table["stage"], self.stage_range.last),
                    )
                )
            
            stage_by_key = {(row["scene_id"], row["query_idx"]): row["stage"] for row in stage_table.to_pylist()}
            
            if self.stage_range is not None:
                # Match scene/query pairs because query_idx values can repeat across scenes.
                query_ids_by_scene: dict[str, list[int]] = {}
                for scene_id, query_idx in stage_by_key:
                    query_ids_by_scene.setdefault(scene_id, []).append(query_idx)
                query_filters = [
                    [("scene_id", "=", scene_id), ("query_idx", "in", query_ids)]
                    for scene_id, query_ids in query_ids_by_scene.items()
                ]

        # Read matching rows in file order; query_idx values remain independent of table positions.
        self._queries = pq.read_table(self.root / "queries" / f"{split}.parquet", filters=query_filters)
        self.query_keys = tuple(zip(self._queries["scene_id"].to_pylist(), self._queries["query_idx"].to_pylist()))
        self._query_positions = {key: position for position, key in enumerate(self.query_keys)}
        self._query_stages = (
            tuple(stage_by_key[key] for key in self.query_keys) if split != "train" else (None,) * len(self)
        )
        self.scene_ids = tuple(sorted({scene_id for scene_id, _ in self.query_keys}))
        self._scene_records = {scene_id: scene_records[scene_id] for scene_id in self.scene_ids}

        # Frame numbers refer to the scene's sampled images, even after selecting fewer queries.
        frame_table = pq.read_table(
            self.root / "frames" / f"{split}.parquet", filters=[("scene_id", "in", list(self.scene_ids))]
        )
        self._frame_names = index_frame_names(frame_table, self._scene_records)

    def __len__(self) -> int:
        """
        Count queries in this reader's selection.

        Returns:
            Number of selected query rows.
        """
        return self._queries.num_rows

    @property
    def query_table(self) -> pa.Table:
        """
        Expose the selected Arrow table for column-oriented processing.

        Returns:
            Selected query rows as an immutable Arrow table.
        """
        return self._queries

    def get_query(self, scene_id: str, query_idx: int) -> QueryRecord:
        """
        Look up a selected query by its stable key. Returned dictionaries are
        fresh values, so caller edits do not modify subsequent lookups.

        Args:
            scene_id: Scene containing the query.
            query_idx: Stable per-scene query identifier.

        Returns:
            The matching query record, including ground-truth target object IDs.
        """
        key = scene_id, query_idx
        position = self._query_positions[key]
        return cast(QueryRecord, self._queries.slice(position, 1).to_pylist()[0])

    def iter_queries(self, batch_size: int = 1024) -> Iterator[QueryRecord]:
        """
        Iterate selected records in query-table order, converting Arrow batches
        to Python dictionaries with at most batch_size records per batch.

        Args:
            batch_size: Maximum rows converted to Python together.

        Returns:
            Iterator of fresh query dictionaries containing the table's values.
        """
        require_integer(batch_size, "batch_size", minimum=1)
        for batch in self._queries.to_batches(max_chunksize=batch_size):
            yield from cast(list[QueryRecord], batch.to_pylist())

    def stage_for(self, scene_id: str, query_idx: int) -> int | None:
        """
        Return the stage assigned to a selected query. Training queries return None.

        Args:
            scene_id: Scene containing the query.
            query_idx: Stable per-scene query identifier.

        Returns:
            One-based stage assignment, or None for training.
        """
        key = scene_id, query_idx
        return self._query_stages[self._query_positions[key]]

    def get_scene(self, scene_id: str) -> SceneRecord:
        """
        Read metadata for a scene represented by selected queries. Counts cover
        all rows stored for the scene and are unchanged by stage selection.

        Args:
            scene_id: A scene in this reader's selection.

        Returns:
            A fresh copy of the scene's record from scenes.json.
        """
        return self._scene_records[scene_id].copy()

    def frame_names(self, scene_id: str) -> tuple[str, ...]:
        """
        Return source filenames in sampled image order. A query's frame indexes this
        tuple; for example, index 1 may hold frame_000010 when sampling every tenth
        source frame. All scene frames are included, even those after the query time.

        Args:
            scene_id: A scene in this reader's selection.

        Returns:
            Frame names ordered by zero-based sampled frame index.
        """
        return self._frame_names[scene_id]

    def get_frame_name(self, scene_id: str, frame_idx: int) -> str:
        """
        Get the source filename at a position in the scene's sampled frame sequence.

        Args:
            scene_id: A scene in this reader's selection.
            frame_idx: Zero-based index on that scene's sampled frame sequence.

        Returns:
            The frame name stored at this index in the frame table.
        """
        return self._frame_names[scene_id][frame_idx]

    def get_annotations(self, scene_id: str) -> SceneAnnotations:
        """
        Read the scene's object visibility histories, including frames after any
        query time. Each call returns a fresh dictionary decoded from its JSON file.

        Args:
            scene_id: Scene represented in this query selection.

        Returns:
            Object labels, visibility intervals, per-frame measurements, and summary statistics.
        """
        scene_record = self._scene_records[scene_id]
        with gzip.open(self.root / scene_record["annotations"], "rt", encoding="utf-8") as stream:
            return cast(SceneAnnotations, json.load(stream))
