"""
Validate annotation files, raw ScanNet++ inputs, and prepared H5 caches before using the readers.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import h5py

from egorecall.config import DatasetPaths
from egorecall.data.annotations import EgoRecallAnnotations
from egorecall.data.integrity import fingerprint_file, relative_file
from egorecall.data.metadata import select_scene_ids
from egorecall.data.records import SceneAnnotations
from egorecall.data.scannetpp import ScanNetPPScene
from egorecall.data.scene_h5 import SceneH5
from egorecall.data.validate_cache import validate_cache_compatibility, validate_scene_cache
from egorecall.data.validate_package import validate_annotation_package
from egorecall.data.validate_sources import validate_object_annotations, validate_source_scene
from egorecall.data.validation import require_integer


@dataclass(frozen=True)
class CheckReport:
    """
    Counts describing the checks that actually ran.

    Args:
        files_verified: Files checked against the dataset manifest.
        queries_checked: Query records validated in the package's full split.
        annotation_scenes: Scene annotation payloads validated.
        source_scenes: Scenes joined to raw geometry and cameras.
        cache_scenes: Prepared scene caches checked for compatibility.
        frames_decoded: Cached RGB-D observations fully decoded and checksummed.
    """

    files_verified: int
    queries_checked: int
    annotation_scenes: int
    source_scenes: int
    cache_scenes: int
    frames_decoded: int


def verify_package(root: Path) -> int:
    """
    Verify manifest byte counts and SHA-256 hashes, requiring coverage of all data
    files used by the reader. File symlinks used by download caches are supported.

    Args:
        root: EgoRecall dataset directory.

    Returns:
        Number of files verified against the manifest.
    """
    with (root / "manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    files = manifest["files"]
    if not isinstance(files, dict) or not files:
        raise ValueError("manifest/files must be a nonempty mapping of filenames to fingerprints.")

    for name, expected in files.items():
        path = relative_file(root, name)
        require_integer(expected["bytes"], f"manifest/files/{name}/bytes")
        actual = fingerprint_file(path)
        if actual != expected:
            raise ValueError(f"{name}: bytes or SHA-256 do not match manifest.json.")

    # A valid checksum list must cover the tables and scene files used for loading.
    split = manifest["selection"]["split"]
    required = {"scenes.json", f"queries/{split}.parquet", f"frames/{split}.parquet"}
    if split != "train":
        required.add(f"stages/{split}.parquet")

    with (root / "scenes.json").open(encoding="utf-8") as stream:
        required.update(scene_record["annotations"] for scene_record in json.load(stream))
    missing = required - files.keys()
    if missing:
        raise ValueError(f"Manifest omits required files: {sorted(missing)}.")
    return len(files)


def check_dataset(
    paths: DatasetPaths,
    *,
    scene_ids: list[str] | None = None,
    check_source: bool = False,
    check_cache: bool = False,
    decode_all: bool = False,
) -> CheckReport:
    """
    Check the complete annotation package and optional source/cache scene joins.
    A scene selection limits source/cache work; package integrity still covers
    every manifest file and all query/annotation records.

    Args:
        paths: Configured dataset locations.
        scene_ids: Optional source/cache scene subset.
        check_source: Validate source camera timelines, object geometry, and annotation joins.
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

    # Validate all annotations before limiting the source/cache work to selected scenes.
    files_verified = verify_package(paths.dataset_root)
    validate_annotation_package(paths.dataset_root)
    with (paths.dataset_root / "manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    annotation_reader = EgoRecallAnnotations(paths.dataset_root, split=manifest["selection"]["split"])
    selected_ids = select_scene_ids(annotation_reader.scene_ids, scene_ids)

    frames_decoded = 0
    for scene_id in selected_ids:
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

    return CheckReport(
        files_verified,
        len(annotation_reader),
        len(annotation_reader.scene_ids),
        len(selected_ids) if check_source else 0,
        len(selected_ids) if check_cache else 0,
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
            paths, scene_id, subsample_factor, 60.0, check_source=True, check_cache=check_cache, decode_all=decode_all
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
