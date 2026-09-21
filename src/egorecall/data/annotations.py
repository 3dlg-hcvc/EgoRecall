"""
Read EgoRecall query tables into Arrow and load scene annotation files on demand.
"""

import gzip
import json
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq

from egorecall.data.metadata import index_frame_names
from egorecall.data.records import QueryKey, QueryRecord, SceneAnnotations, SceneRecord, Split, decode_program
from egorecall.data.schema import FRAME_SCHEMA, QUERY_SCHEMA, STAGE_SCHEMA, validate_table
from egorecall.data.stages import StageRange, parse_stages, require_stage_range
from egorecall.data.validation import require_integer, require_text, validate_annotations, validate_scene


class EgoRecallAnnotations:
    """
    Read query tables and scene annotations for a split and optional stage selection.
    Stable query keys are independent of table positions. Queries are loaded
    into Arrow before stage selection; scene annotation files are read on demand.
    Validation checks table schemas, scene/frame joins, stage membership, and
    selected query fields.

    The supported layout has a single-split manifest and one Parquet file per
    table. Validation covers record structure and joins; it does not verify
    the file checksums listed in the manifest.

    Args:
        dataset_root: EgoRecall data directory containing manifest.json.
        split: Requested train, val, or test split, matching the dataset manifest.
        stages: Exact stage number, inclusive LO:HI range, or None for all queries
            present in the split's table. Training must be selected without stages.
    """

    def __init__(self, dataset_root: Path, split: str = "test", stages: int | str | None = None) -> None:
        """
        Load query, stage, and frame tables, validate their joins, and index
        selected query keys. Scene annotation files are read by get_annotations().

        Args:
            dataset_root: Local EgoRecall data directory containing manifest.json.
            split: One available benchmark split.
            stages: Exact stage, inclusive range, or all rows in the table with None.
        """
        if not isinstance(dataset_root, Path):
            raise TypeError("dataset_root must be a pathlib.Path.")
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split {split!r}; use train, val, or test.")

        self.root = dataset_root.expanduser().resolve()
        self.split = cast(Split, split)
        self.stage_range: StageRange | None = parse_stages(stages) if stages is not None else None

        # Validate the package format before loading its scene and query records.
        manifest = self._read_json("manifest.json")
        if not isinstance(manifest, dict):
            raise ValueError("manifest.json must contain a JSON object.")
        schema_version = require_integer(manifest["schema_version"], "manifest/schema_version", minimum=1)
        if schema_version != 1:
            raise ValueError(f"Unsupported package schema_version: {schema_version}.")
        self._manifest = manifest

        # Load the split's full query table before selecting stages.
        scenes = self._load_scenes()
        queries = self._read_table("queries", QUERY_SCHEMA)
        query_keys = self._query_keys(queries, "queries")
        self._validate_scene_counts(queries, query_keys, scenes)

        # Join stage assignments by keys; the downloaded subset may have large ID gaps.
        stage_by_key = self._load_stages(query_keys)
        selected: range | list[int]
        if self.stage_range is None:
            selected = range(len(query_keys))
        else:
            selected = [
                position
                for position, key in enumerate(query_keys)
                if self.stage_range.first <= stage_by_key[key] <= self.stage_range.last
            ]

        # Reuse the immutable Arrow table when the selection includes every row.
        if len(selected) == queries.num_rows:
            self._queries = queries
            self.query_keys = tuple(query_keys)
            self._stage_by_key = stage_by_key
        else:
            self._queries = queries.take(pa.array(selected, type=pa.int64()))
            self.query_keys = tuple(query_keys[position] for position in selected)
            self._stage_by_key = {key: stage_by_key[key] for key in self.query_keys}

        # Index the selected queries and the scenes represented by them.
        self._query_positions = {key: position for position, key in enumerate(self.query_keys)}
        self.scene_ids = tuple(sorted({key[0] for key in self.query_keys}))
        self._scenes = {scene_id: scenes[scene_id] for scene_id in self.scene_ids}

        # Frame names are indexed explicitly, even if Parquet rows are reordered.
        frames = self._read_table("frames", FRAME_SCHEMA)
        all_frame_names = index_frame_names(frames, scenes)
        self._frame_names = {scene_id: all_frame_names[scene_id] for scene_id in self.scene_ids}
        self._validate_manifest_counts(queries.num_rows, len(stage_by_key), frames.num_rows, len(scenes))

        # Validate selected queries in batches and collect each scene's target IDs.
        self._target_ids: dict[str, set[int]] = defaultdict(set)
        for query in self.iter_queries():
            self._validate_query(query)
            self._target_ids[query["scene_id"]].update(query["target_oids"])

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
        key = self._selected_key(scene_id, query_idx)
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
        key = self._selected_key(scene_id, query_idx)
        if self.split == "train":
            return None
        return self._stage_by_key[key]

    def get_scene(self, scene_id: str) -> SceneRecord:
        """
        Read metadata for a scene represented by selected queries. Counts cover
        all rows stored for the scene and are unchanged by stage selection.

        Args:
            scene_id: A scene in this reader's selection.

        Returns:
            A fresh copy of the scene's record from scenes.json.
        """
        if scene_id not in self._scenes:
            raise KeyError(f"Scene {scene_id!r} is not in the selected {self.split} queries.")
        return self._scenes[scene_id].copy()

    def frame_names(self, scene_id: str) -> tuple[str, ...]:
        """
        Return all frame names for a selected scene, ordered by canonical index.
        The mapping covers the full timeline, including frames after a query's time.

        Args:
            scene_id: A scene in this reader's selection.

        Returns:
            Frame names ordered by zero-based canonical frame index.
        """
        return self._frame_names[scene_id]

    def get_frame_name(self, scene_id: str, frame_idx: int) -> str:
        """
        Look up the source frame name for a canonical index using the frame table.

        Args:
            scene_id: A scene in this reader's selection.
            frame_idx: Zero-based index on that scene's canonical timeline.

        Returns:
            The frame name stored at this index in the frame table.
        """
        names = self.frame_names(scene_id)
        require_integer(frame_idx, "frame_idx")
        if frame_idx >= len(names):
            raise IndexError(f"{scene_id}: frame {frame_idx} is outside the {len(names)}-frame timeline.")
        return names[frame_idx]

    def get_annotations(self, scene_id: str) -> SceneAnnotations:
        """
        Load and validate a selected scene's full annotations on demand. All
        filtered objects and the complete timeline are retained. Load once per
        scene for repeated use; this method returns a fresh decoded record.

        Args:
            scene_id: A scene in this reader's selection.

        Returns:
            Full-scene supervision, including observations after individual queries.
        """
        scene = self._scenes[scene_id]
        path = self._package_path(scene["annotations"])
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            annotation = validate_annotations(json.load(stream), scene)

        # Confirm that every target ID in the selected queries has an object annotation.
        object_ids = {int(oid) for oid in annotation["objects"]}
        missing = self._target_ids[scene_id] - object_ids
        if missing:
            raise ValueError(f"{scene_id}: target IDs have no annotation: {sorted(missing)}.")
        return annotation

    def _package_path(self, relative_path: str) -> Path:
        """
        Require metadata paths relative to dataset_root. File symlinks are allowed,
        including cached downloads whose files link to an external blob store.

        Args:
            relative_path: Path from scene metadata or the table layout, relative to dataset_root.

        Returns:
            Absolute lexical path under dataset_root, without resolving file symlinks.
        """
        path = Path(relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Package path must stay within the dataset directory: {relative_path!r}.")
        return self.root / path

    def _read_json(self, relative_path: str) -> object:
        """
        Decode a dataset JSON file before validating its specific record type.

        Args:
            relative_path: Filename relative to dataset_root.

        Returns:
            The decoded value, whose structure is checked by the calling loader.
        """
        with self._package_path(relative_path).open(encoding="utf-8") as stream:
            return json.load(stream)

    def _read_table(self, name: str, schema: pa.Schema) -> pa.Table:
        """
        Read one split table and reject incompatible columns or null fields.

        Args:
            name: Table directory, such as queries or frames.
            schema: Required Arrow column names and types.

        Returns:
            The table with values and row order preserved from the Parquet file.
        """
        path = self._package_path(f"{name}/{self.split}.parquet")
        if not path.is_file():
            raise FileNotFoundError(f"Required {self.split} table is missing: {path}.")

        table = pq.read_table(path)
        validate_table(table, schema, name)
        return table

    def _load_scenes(self) -> dict[str, SceneRecord]:
        """
        Load scenes.json and reject duplicate identities or an
        unavailable split before attempting table reads.

        Returns:
            Scene records belonging to the requested split.
        """
        records = self._read_json("scenes.json")
        if not isinstance(records, list):
            raise ValueError("scenes.json must contain a list of scene records.")

        scenes: dict[str, SceneRecord] = {}
        for value in records:
            scene = validate_scene(value)
            scene_id = scene["scene_id"]
            if scene_id in scenes:
                raise ValueError(f"Duplicate scene metadata: {scene_id}.")
            self._package_path(scene["annotations"])
            scenes[scene_id] = scene

        self.available_splits = tuple(
            split for split in ("train", "val", "test") if any(scene["split"] == split for scene in scenes.values())
        )
        if self.split not in self.available_splits:
            raise ValueError(f"Split {self.split!r} is not packaged; available splits: {self.available_splits}.")
        return {scene_id: scene for scene_id, scene in scenes.items() if scene["split"] == self.split}

    def _query_keys(self, table: pa.Table, name: str) -> list[QueryKey]:
        """
        Extract keys from a query or stage table. Require unique scene/query ID
        pairs, valid identifiers, and membership in the requested split.

        Args:
            table: Schema-validated queries or stages table.
            name: Table description used in errors.

        Returns:
            Stable keys in table row order.
        """
        if any(split != self.split for split in table["split"].to_pylist()):
            raise ValueError(f"{name}: rows disagree with the requested split {self.split!r}.")

        keys = list(zip(table["scene_id"].to_pylist(), table["query_idx"].to_pylist(), strict=True))
        if len(keys) != len(set(keys)):
            raise ValueError(f"{name}: duplicate (scene_id, query_idx) keys.")
        for scene_id, query_idx in keys:
            require_text(scene_id, f"{name}/scene_id")
            require_integer(query_idx, f"{name}/query_idx")
        return keys

    def _validate_scene_counts(self, table: pa.Table, keys: list[QueryKey], scenes: dict[str, SceneRecord]) -> None:
        """
        Check scene membership and per-scene query counts against scenes.json
        before applying the stage selection.

        Args:
            table: Full requested query split.
            keys: Its stable row keys.
            scenes: Metadata for all scenes in the split.
        """
        counts = Counter(scene_id for scene_id, _ in keys)
        if set(counts) - set(scenes):
            raise ValueError("queries: scene IDs are absent from the split's scene metadata.")
        if table.num_rows == 0 or any(counts[scene_id] != scene["num_queries"] for scene_id, scene in scenes.items()):
            raise ValueError("queries: row counts disagree with scenes.json or the split is empty.")

    def _load_stages(self, query_keys: list[QueryKey]) -> dict[QueryKey, int]:
        """
        Join stage assignments to the query table and validate the requested
        range against the stage numbers present in the assignment table.

        Args:
            query_keys: All query keys in the split's table, before stage selection.

        Returns:
            One stage per query key, or an empty mapping for unstaged training.
        """
        if self.split == "train":
            if self.stage_range is not None:
                raise ValueError("Training is unstaged; omit stages when reading train.")
            if self._package_path("stages/train.parquet").exists():
                raise ValueError("Training must be unstaged; found stages/train.parquet.")
            self.available_stages: tuple[int, ...] = ()
            return {}

        table = self._read_table("stages", STAGE_SCHEMA)
        keys = self._query_keys(table, "stages")
        if set(keys) != set(query_keys):
            raise ValueError("Query/stage membership differs; every packaged query needs one assignment.")

        stages = table["stage"].to_pylist()
        for stage in stages:
            require_integer(stage, "stages/stage", minimum=1)
        self.available_stages = tuple(sorted(set(stages)))

        # Require every stage in the requested range to be present in the assignment table.
        if self.stage_range is not None:
            require_stage_range(self.stage_range, self.available_stages)
        return dict(zip(keys, stages, strict=True))

    def _validate_manifest_counts(self, queries: int, stage_assignments: int, frames: int, scenes: int) -> None:
        """
        Compare table counts and stage assignments with the single-split manifest.
        Require every manifest field used by this check and reject mismatched
        splits, counts, or stage ranges.

        Args:
            queries: Number of query-table rows before stage selection.
            stage_assignments: Number of query-to-stage assignment rows.
            frames: Number of frame-mapping rows.
            scenes: Number of scenes in the split.
        """
        selection = self._manifest["selection"]
        if not isinstance(selection, dict):
            raise ValueError("manifest/selection must contain a JSON object.")
        if selection["split"] != self.split:
            raise ValueError(f"Manifest selection does not match the requested {self.split} split.")

        counts = self._manifest["counts"]
        if not isinstance(counts, dict):
            raise ValueError("manifest/counts must contain a JSON object.")
        expected = {"queries": queries, "stage_assignments": stage_assignments, "frames": frames, "scenes": scenes}
        for name, observed in expected.items():
            declared = require_integer(counts[name], f"manifest/counts/{name}")
            if declared != observed:
                raise ValueError(f"manifest/counts/{name}: declared {declared}, found {observed}.")

        if self.split != "train":
            first = require_integer(selection["stage_from"], "manifest/stage_from", minimum=1)
            last = require_integer(selection["stage_to"], "manifest/stage_to", minimum=first)
            if last - first + 1 != len(self.available_stages) or self.available_stages != tuple(range(first, last + 1)):
                raise ValueError("Packaged stages disagree with the range declared in manifest.json.")

    def _validate_query(self, query: QueryRecord) -> None:
        """
        Check query time against the scene's frame count, validate the target-ID
        partition, and check the nested program representation.

        Args:
            query: One selected record from the schema-validated Arrow table.
        """
        scene_id, query_idx = query["scene_id"], query["query_idx"]
        context = f"{scene_id}/{query_idx}"
        frame = require_integer(query["frame"], f"{context}/frame")
        if frame >= self._scenes[query["scene_id"]]["num_frames"]:
            raise ValueError(f"{context}: query frame is outside the canonical timeline.")

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
            decode_program(query["program_json"])
        except ValueError as error:
            raise ValueError(f"{context}: invalid program_json: {error}") from error

    def _selected_key(self, scene_id: str, query_idx: int) -> QueryKey:
        """
        Require a valid scene/query ID pair belonging to this reader's selection.

        Args:
            scene_id: Scene identifier.
            query_idx: Stable per-scene integer query identifier.

        Returns:
            The key after confirming it belongs to this selection.
        """
        require_text(scene_id, "scene_id")
        require_integer(query_idx, "query_idx")
        key = scene_id, query_idx
        if key not in self._query_positions:
            raise KeyError(f"Query {key} is not in the selected {self.split} queries.")
        return key
