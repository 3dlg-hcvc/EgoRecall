"""
Access query text and past frames for a method, and read answers separately for evaluation.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property
from types import TracebackType

from egorecall.arguments import require_integer
from egorecall.config import DatasetPaths
from egorecall.data.annotations import EgoRecallAnnotations
from egorecall.data.records import QueryRecord, SceneAnnotations
from egorecall.data.scene_h5 import FrameCamera, Observation, SceneH5
from egorecall.geometry.boxes import ObjectGeometry


@dataclass(frozen=True)
class QueryInput:
    """
    Query fields available to a method at query time.

    Args:
        scene_id: Scene containing the observation history.
        query_idx: Stable per-scene query identifier.
        description: Natural-language object reference.
        frame: Zero-based number of the last frame the method may observe.
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
        scene_h5: Open scene H5 file.
        through_frame: Last sampled frame available to the method.
    """

    def __init__(self, scene_h5: SceneH5, through_frame: int) -> None:
        """
        Keep only the frame indices through the query time. Indexing this shorter
        sequence prevents future-frame access, including through negative indices.

        Args:
            scene_h5: Open scene H5 file.
            through_frame: Zero-based number of the last frame available to the method.
        """
        require_integer(through_frame, "through_frame")
        if through_frame >= len(scene_h5.frame_names):
            raise IndexError("Query cutoff is outside the observation timeline.")

        self._scene_h5 = scene_h5
        self._frame_names = scene_h5.frame_names[: through_frame + 1]
        self._frame_indices = range(len(scene_h5.frame_names))[: through_frame + 1]

    @property
    def frame_names(self) -> tuple[str, ...]:
        """
        List source names through the query cutoff.

        Returns:
            Immutable names ordered by sampled frame index.
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
            frame_idx: Zero-based sampled frame index within this history.

        Returns:
            One observation with no target labels, programs, or visibility history.
        """
        return self._scene_h5.observation(self._frame_indices[frame_idx])

    def camera(self, frame_idx: int) -> FrameCamera:
        """
        Read an available frame's camera metadata without decoding its images.

        Args:
            frame_idx: Zero-based sampled frame index within this query's history.

        Returns:
            Frame identity, timestamp, pose, and RGB/depth intrinsics with fresh arrays.
        """
        return self._scene_h5.camera(self._frame_indices[frame_idx])

    def encoded_image(self, frame_idx: int, kind: str = "rgb") -> bytes:
        """
        Read an encoded image within the same query-time boundary.

        Args:
            frame_idx: Zero-based sampled frame index within this history.
            kind: One of rgb, depth, or mask.

        Returns:
            JPEG or PNG bytes for the requested observation.
        """
        return self._scene_h5.encoded_image(self._frame_indices[frame_idx], kind)

    def __iter__(self) -> Iterator[Observation]:
        """
        Decode the available history in order, retaining one observation at a time.

        Returns:
            Iterator from frame zero through the query frame.
        """
        for frame_idx in range(len(self.frame_names)):
            yield self._scene_h5.observation(frame_idx)


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
    Full-scene ground truth for inspection or evaluation.

    Args:
        annotations: Filtered object visibility over the complete timeline.
        source_objects: Every object in the source ScanNet++ annotation.
        filtered_objects: Source geometry for the objects retained by EgoRecall.
    """

    annotations: SceneAnnotations
    source_objects: dict[int, ObjectGeometry]
    filtered_objects: dict[int, ObjectGeometry]


class EgoRecallScene:
    """
    Own one scene cache and load its visibility supervision only when requested.
    Pass query() results to methods; answer() and supervision contain ground truth.

    Args:
        paths: Dataset and local cache locations.
        annotation_reader: Query selection containing this scene.
        scene_id: Scene represented by at least one selected query.
    """

    def __init__(self, paths: DatasetPaths, annotation_reader: EgoRecallAnnotations, scene_id: str) -> None:
        """
        Open this scene's H5 file. Annotation and cache consistency is checked by egorecall-check.

        Args:
            paths: Dataset and cache locations.
            annotation_reader: Query selection containing this scene.
            scene_id: Selected scene.
        """
        if paths.cache_root is None:
            raise ValueError("cache_root is required to read observations.")

        self.scene_id = scene_id
        self._annotation_reader = annotation_reader
        self._scene_h5 = SceneH5(paths.cache_root / f"{scene_id}.h5")

    def query(self, query_idx: int) -> QuerySample:
        """
        Return the query text and frames through its query time, without answer IDs or visibility histories.

        Args:
            query_idx: Stable identifier within this scene and dataset selection.

        Returns:
            QueryInput plus an ObservationWindow containing frames 0 through the query frame.
        """
        query_record = self._annotation_reader.get_query(self.scene_id, query_idx)
        query = QueryInput(self.scene_id, query_idx, query_record["description"], query_record["frame"])
        return QuerySample(query, ObservationWindow(self._scene_h5, query.frame))

    def answer(self, query_idx: int) -> QueryRecord:
        """
        Read a query's full record for supervision, including targets and DSL program.

        Args:
            query_idx: Stable identifier within this scene and dataset selection.

        Returns:
            A fresh query dictionary containing its ground-truth answer.
        """
        return self._annotation_reader.get_query(self.scene_id, query_idx)

    @cached_property
    def supervision(self) -> SceneSupervision:
        """
        Read visibility histories and cached boxes, then select boxes by the annotated
        object IDs. This includes future visibility and must not be passed to methods.

        Returns:
            Scene visibility records, all source objects, and the subset retained by the visibility filter.
        """
        scene_annotations = self._annotation_reader.get_annotations(self.scene_id)
        source_objects = self._scene_h5.objects()
        filtered_objects = {
            int(object_id): source_objects[int(object_id)] for object_id in scene_annotations["objects"]
        }
        return SceneSupervision(scene_annotations, source_objects, filtered_objects)

    def close(self) -> None:
        """
        Close the observation cache and invalidate further image access through its windows.
        """
        self._scene_h5.close()

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
