"""
Validate annotation files, raw ScanNet++ inputs, and prepared H5 caches before using the readers.
"""

from dataclasses import dataclass

import h5py

from egorecall.config import DatasetPaths
from egorecall.data.annotations import EgoRecallAnnotations, read_scene_records, select_scene_ids
from egorecall.data.records import SceneAnnotations
from egorecall.data.scannetpp import SOURCE_FPS, ScanNetPPScene
from egorecall.data.scene_h5 import SceneH5
from egorecall.validation.cache import validate_cache_compatibility, validate_scene_cache
from egorecall.validation.package import validate_annotation_package, verify_package
from egorecall.validation.sources import validate_object_annotations, validate_source_scene


@dataclass(frozen=True)
class CheckReport:
    """
    Counts describing the checks that actually ran.

    Args:
        files_verified: Files checked against the dataset manifest.
        queries_checked: Query records validated in the package's splits.
        annotation_scenes: Scene annotation payloads validated.
        source_scenes: Scenes whose ScanNet++ camera and object records were checked.
        cache_scenes: Prepared scene caches checked for compatibility.
        frames_decoded: Cached RGB-D observations fully decoded and checksummed.
    """

    files_verified: int
    queries_checked: int
    annotation_scenes: int
    source_scenes: int
    cache_scenes: int
    frames_decoded: int


def check_dataset(
    paths: DatasetPaths,
    *,
    split: str | None = None,
    stages: int | str | None = None,
    scene_ids: list[str] | None = None,
    check_source: bool = False,
    check_cache: bool = False,
    decode_all: bool = False,
) -> CheckReport:
    """
    Check all EgoRecall annotations and optionally their ScanNet++ source files and prepared H5 caches.
    A split, stage, or scene selection limits source/cache work and chooses scenes the same way as
    egorecall-prepare; package integrity still covers every manifest file and all query/annotation records.

    Args:
        paths: Configured dataset locations.
        split: Split whose scenes receive source/cache checks, or None for scenes from every split.
        stages: Exact stage or inclusive LO:HI range within split, selecting scenes with queries in those stages.
        scene_ids: Optional source/cache scene subset, taken from the split and stage selection when one is given.
        check_source: Check source cameras and boxes, and compare frame names and object IDs/labels with annotations.
        check_cache: Require and validate a prepared cache for each selected scene.
        decode_all: Decode every cached frame; otherwise check the first and last.

    Returns:
        Counts for package, source, cache, and decoded-frame checks.
    """
    if paths.dataset_root is None:
        raise ValueError("Dataset checks require dataset_root.")
    if check_source and paths.scannetpp_root is None:
        raise ValueError("Source checks require scannetpp_root.")
    if check_cache and paths.cache_root is None:
        raise ValueError("Cache checks require cache_root.")
    if decode_all and not check_cache:
        raise ValueError("decode_all requires cache checking.")
    if stages is not None and split is None:
        raise ValueError("Stage selection requires a split.")
    if (split is not None or scene_ids is not None) and not (check_source or check_cache):
        raise ValueError("Split, stage, and scene selections apply only to source or cache checks.")

    # Validate all annotations before limiting the source/cache work to selected scenes.
    files_verified = verify_package(paths.dataset_root)
    split_counts = validate_annotation_package(paths.dataset_root)

    # Select scenes as preparation does: from one split and its stages, or by scene ID across every split.
    requested_by_split: dict[str, list[str] | None] = {}
    if split is not None:
        requested_by_split[split] = scene_ids
    elif check_source or check_cache:
        scene_records = read_scene_records(paths.dataset_root)
        selected_ids = select_scene_ids(tuple(sorted(scene_records)), scene_ids)
        for split_name in split_counts:
            split_scene_ids = [scene_id for scene_id in selected_ids if scene_records[scene_id]["split"] == split_name]
            if split_scene_ids:
                requested_by_split[split_name] = split_scene_ids

    # Compare each selected scene's source files or prepared cache with its annotations, one split at a time.
    scenes_checked = 0
    frames_decoded = 0
    for split_name, requested_ids in requested_by_split.items():
        annotation_reader = EgoRecallAnnotations(paths.dataset_root, split=split_name, stages=stages)
        for scene_id in select_scene_ids(annotation_reader.scene_ids, requested_ids):
            scene_record = annotation_reader.get_scene(scene_id)
            frames_decoded += _check_scene_assets(
                paths,
                scene_id,
                scene_record["subsample_factor"],
                scene_record["source_fps"],
                frame_names=annotation_reader.frame_names(scene_id),
                scene_annotations=annotation_reader.get_annotations(scene_id),
                check_source=check_source,
                check_cache=check_cache,
                decode_all=decode_all,
            )
            scenes_checked += 1

    return CheckReport(
        files_verified,
        sum(counts["queries"] for counts in split_counts.values()),
        sum(counts["scenes"] for counts in split_counts.values()),
        scenes_checked if check_source else 0,
        scenes_checked if check_cache else 0,
        frames_decoded,
    )


def check_source_scenes(
    paths: DatasetPaths,
    scene_ids: list[str],
    subsample_factor: int,
    *,
    check_cache: bool = False,
    decode_all: bool = False,
) -> CheckReport:
    """
    Check raw inputs, and optionally their prepared caches, without an annotation package.

    Args:
        paths: Source and optional cache directories.
        scene_ids: ScanNet++ scenes to check.
        subsample_factor: Sampling stride used for preparation.
        check_cache: Also compare each H5 cache with its source files.
        decode_all: Decode every cached frame rather than only the first and last.

    Returns:
        Source/cache check counts; annotation-related counts are zero.
    """
    if paths.scannetpp_root is None:
        raise ValueError("Source checks require scannetpp_root.")
    if check_cache and paths.cache_root is None:
        raise ValueError("Cache checks require cache_root.")
    if decode_all and not check_cache:
        raise ValueError("decode_all requires cache checking.")
    if not scene_ids or len(scene_ids) != len(set(scene_ids)):
        raise ValueError("Supply a nonempty list of distinct scene IDs.")

    frames_decoded = sum(
        _check_scene_assets(
            paths,
            scene_id,
            subsample_factor,
            SOURCE_FPS,
            check_source=True,
            check_cache=check_cache,
            decode_all=decode_all,
        )
        for scene_id in scene_ids
    )
    return CheckReport(0, 0, 0, len(scene_ids), len(scene_ids) if check_cache else 0, frames_decoded)


def _check_scene_assets(
    paths: DatasetPaths,
    scene_id: str,
    subsample_factor: int,
    source_fps: float,
    *,
    frame_names: tuple[str, ...] | None = None,
    scene_annotations: SceneAnnotations | None = None,
    check_source: bool,
    check_cache: bool,
    decode_all: bool,
) -> int:
    """
    Compare one scene's source or cache with the expected frame names and object labels.

    Args:
        paths: Configured source and cache directories.
        scene_id: Scene to check.
        subsample_factor: Expected pose-record sampling stride.
        source_fps: Expected nominal source frame rate.
        frame_names: Annotation frame names, or None when checking source data alone.
        scene_annotations: Object visibility data, or None without an annotation package.
        check_source: Inspect raw source files and camera/box values.
        check_cache: Inspect the prepared H5 file and all image checksums.
        decode_all: Decode all cached images, in addition to checking their headers.

    Returns:
        Number of cached frames decoded.
    """
    source_cameras = None
    source_objects = None
    source_files = None
    if check_source:
        source_scene = ScanNetPPScene(paths.scannetpp_root, scene_id)
        source_cameras, source_objects = validate_source_scene(source_scene, subsample_factor)
        if frame_names is not None and source_cameras.frame_names != frame_names:
            raise ValueError(f"{scene_id}: source frame names differ from the annotation frame mapping.")
        frame_names = source_cameras.frame_names
        if scene_annotations is not None:
            validate_object_annotations(scene_id, source_objects, scene_annotations)
        if check_cache:
            source_files = source_scene.cache_fingerprints()

    if not check_cache:
        return 0

    cache_path = paths.cache_root / f"{scene_id}.h5"
    frames_decoded = validate_scene_cache(cache_path, decode_all=decode_all)
    with h5py.File(cache_path, "r") as h5_file:
        validate_cache_compatibility(
            h5_file,
            scene_id,
            frame_names,
            subsample_factor,
            source_fps,
            cameras=source_cameras,
            source_files=source_files,
            objects_by_id=source_objects,
        )
    if scene_annotations is not None:
        with SceneH5(cache_path) as scene_h5:
            validate_object_annotations(scene_id, scene_h5.objects(), scene_annotations)
    return frames_decoded
