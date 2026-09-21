"""
Join benchmark queries, bounded observation histories, and separate scene supervision.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property
from types import TracebackType

from egorecall.config import DatasetPaths
from egorecall.data.annotations import EgoRecallAnnotations
from egorecall.data.records import QueryRecord, SceneAnnotations
from egorecall.data.scannetpp import ObjectGeometry
from egorecall.data.scene_h5 import FrameCamera, Observation, SceneH5
from egorecall.data.validation import require_integer


@dataclass(frozen=True)
class QueryInput:
    """
    Query fields available to a method at query time.

    Args:
        scene_id: Scene containing the observation history.
        query_idx: Stable per-scene query identifier.
        description: Natural-language object reference.
        frame: Inclusive canonical observation cutoff.
    """

    scene_id: str
    query_idx: int
    description: str
    frame: int


class ObservationWindow:
    """
    Access frames through an inclusive query-time cutoff. The owning EgoRecallScene
    context must remain open while this window is used.

    Args:
        cache: Open observation cache.
        through_frame: Last canonical frame available to the method.
    """

    def __init__(self, cache: SceneH5, through_frame: int) -> None:
        """
        Bound the accessible timeline without reading image payloads.

        Args:
            cache: Open observation cache.
            through_frame: Inclusive canonical cutoff.
        """
        require_integer(through_frame, "through_frame")
        if through_frame >= len(cache.frame_names):
            raise IndexError("Query cutoff is outside the observation timeline.")

        self._cache = cache
        self._frame_names = cache.frame_names[: through_frame + 1]

    @property
    def frame_names(self) -> tuple[str, ...]:
        """
        List source names through the query cutoff.

        Returns:
            Immutable names ordered by canonical index.
        """
        return self._frame_names

    def __len__(self) -> int:
        """
        Count frames available at query time.

        Returns:
            Number of frames through the inclusive cutoff.
        """
        return len(self.frame_names)

    def frame(self, frame_idx: int) -> Observation:
        """
        Decode an available observation, rejecting access beyond the query cutoff.

        Args:
            frame_idx: Zero-based canonical index within this history.

        Returns:
            One observation with no target labels, programs, or visibility history.
        """
        self._require_frame(frame_idx)
        return self._cache.observation(frame_idx)

    def camera(self, frame_idx: int) -> FrameCamera:
        """
        Read an available frame's camera metadata without decoding its images.

        Args:
            frame_idx: Zero-based canonical index within this query's history.

        Returns:
            Frame identity, timestamp, pose, and RGB/depth intrinsics with fresh arrays.
        """
        self._require_frame(frame_idx)
        return self._cache.camera(frame_idx)

    def encoded_image(self, frame_idx: int, kind: str = "rgb") -> bytes:
        """
        Read an encoded image within the same query-time boundary.

        Args:
            frame_idx: Zero-based canonical index within this history.
            kind: One of rgb, depth, or mask.

        Returns:
            JPEG or PNG bytes for the requested observation.
        """
        self._require_frame(frame_idx)
        return self._cache.encoded_image(frame_idx, kind)

    def _require_frame(self, frame_idx: int) -> None:
        """
        Reject invalid indices and future observations without clamping them.

        Args:
            frame_idx: Requested canonical index.
        """
        require_integer(frame_idx, "frame_idx")
        if frame_idx >= len(self.frame_names):
            raise IndexError(f"Frame {frame_idx} exceeds the query cutoff {len(self.frame_names) - 1}.")

    def __iter__(self) -> Iterator[Observation]:
        """
        Decode the available history in order, retaining one observation at a time.

        Returns:
            Iterator from frame zero through the query frame.
        """
        for frame_idx in range(len(self.frame_names)):
            yield self._cache.observation(frame_idx)


@dataclass(frozen=True)
class QuerySample:
    """
    Method-facing query and its bounded observation history.

    Args:
        query: Query identity, text, and time only.
        observations: History through the query frame, inclusive.
    """

    query: QueryInput
    observations: ObservationWindow


@dataclass(frozen=True)
class SceneSupervision:
    """
    Full-scene ground truth for generation, inspection, or evaluation.

    Args:
        annotations: Filtered object visibility over the complete timeline.
        source_objects: Every object in the source ScanNet++ annotation.
        filtered_objects: Source geometry for the objects retained by EgoRecall.
    """

    annotations: SceneAnnotations
    source_objects: dict[int, ObjectGeometry]
    filtered_objects: dict[int, ObjectGeometry]


def join_supervision(
    scene_id: str, objects: dict[int, ObjectGeometry], annotations: SceneAnnotations
) -> SceneSupervision:
    """
    Join filtered visibility annotations to the complete source-object population.
    Require matching object IDs and labels; missing objects are data errors.

    Args:
        scene_id: Scene supplying geometry.
        objects: Full source-object population from a scene cache or raw source reader.
        annotations: Full-scene EgoRecall visibility annotations.

    Returns:
        Separate source and filtered object mappings with shared geometry records.
    """
    if annotations["scene_id"] != scene_id:
        raise ValueError("Source scene and annotation scene do not match.")

    filtered: dict[int, ObjectGeometry] = {}
    for key, annotation in annotations["objects"].items():
        oid = int(key)
        obj = objects[oid]
        if obj.label != annotation["label"]:
            raise ValueError(f"{scene_id}/{oid}: source and annotation labels differ.")
        filtered[oid] = obj

    return SceneSupervision(annotations, objects, filtered)


class EgoRecallScene:
    """
    Own one scene cache and load its visibility supervision only when requested.
    Pass query() results to methods; answer() and supervision contain ground truth.

    Args:
        paths: Dataset and local cache locations.
        annotations: Query selection containing this scene.
        scene_id: Scene represented by at least one selected query.
    """

    def __init__(self, paths: DatasetPaths, annotations: EgoRecallAnnotations, scene_id: str) -> None:
        """
        Open and validate a prepared scene against the annotation frame mapping.

        Args:
            paths: Dataset and cache locations.
            annotations: Query selection containing this scene.
            scene_id: Selected scene.
        """
        metadata = annotations.get_scene(scene_id)
        if paths.cache_root is None:
            raise ValueError("cache_root is required to read observations.")

        self.scene_id = scene_id
        self._annotations = annotations

        # Keep the handle only if the cache matches this scene's annotation timeline.
        self._cache = SceneH5(paths.cache_root / f"{scene_id}.h5")
        try:
            self._cache.validate_compatibility(
                scene_id, annotations.frame_names(scene_id), metadata["subsample_factor"], metadata["source_fps"]
            )
        except BaseException:
            self._cache.close()
            raise

    def query(self, query_idx: int) -> QuerySample:
        """
        Select the query inputs and legal history without exposing supervision.

        Args:
            query_idx: Stable identifier within this scene and dataset selection.

        Returns:
            Text/time/identity and observations through the query frame.
        """
        record = self._annotations.get_query(self.scene_id, query_idx)
        query = QueryInput(self.scene_id, query_idx, record["description"], record["frame"])
        return QuerySample(query, ObservationWindow(self._cache, query.frame))

    def answer(self, query_idx: int) -> QueryRecord:
        """
        Read a query's full record for supervision, including targets and DSL program.

        Args:
            query_idx: Stable identifier within this scene and dataset selection.

        Returns:
            A fresh query dictionary containing its ground-truth answer.
        """
        return self._annotations.get_query(self.scene_id, query_idx)

    @cached_property
    def supervision(self) -> SceneSupervision:
        """
        Join full-scene visibility with cached source geometry once for this scene context.
        The retained record includes future visibility and must not be passed to methods.

        Returns:
            Scene annotations and explicit source/filtered object populations.
        """
        return join_supervision(self.scene_id, self._cache.objects(), self._annotations.get_annotations(self.scene_id))

    def close(self) -> None:
        """
        Close the observation cache and invalidate further image access through its windows.
        """
        self._cache.close()

    def __enter__(self) -> EgoRecallScene:
        """
        Enter this scene context.

        Returns:
            This open scene accessor.
        """
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        """
        Close the observation cache on normal exit or failure.

        Args:
            exc_type: Active exception type, if any.
            exc: Active exception, if any.
            traceback: Active exception traceback, if any.
        """
        self.close()


class EgoRecallDataset:
    """
    Access benchmark annotations and prepared scenes through configured data locations.
    The annotations reader defines the split/stage selection; open_scene() provides
    a scene context for query-time observations and separate supervision.

    Args:
        paths: Configured locations, including dataset_root for the annotation package.
        split: Benchmark split present in the annotation package.
        stages: Exact stage, inclusive range, or None for all stored queries.
    """

    def __init__(self, paths: DatasetPaths, split: str = "test", stages: int | str | None = None) -> None:
        """
        Load query metadata; scene caches and geometry are opened separately.

        Args:
            paths: Configured locations, including dataset_root for the annotation package.
            split: Benchmark split.
            stages: Optional query-stage selection.
        """
        if paths.dataset_root is None:
            raise ValueError("dataset_root is required to load EgoRecall annotations.")

        self.paths = paths
        self.annotations = EgoRecallAnnotations(paths.dataset_root, split=split, stages=stages)

    def open_scene(self, scene_id: str) -> EgoRecallScene:
        """
        Open a prepared scene for query access and optional supervision.

        Args:
            scene_id: Scene represented by the query selection.

        Returns:
            Scene context manager; close it after consuming its observation windows.
        """
        return EgoRecallScene(self.paths, self.annotations, scene_id)
