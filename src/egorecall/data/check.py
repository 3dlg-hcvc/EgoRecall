"""
Check annotation-package integrity and joins to raw sources and prepared observations.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import pyarrow.compute as pc

from egorecall.config import DatasetPaths
from egorecall.data.annotations import EgoRecallAnnotations
from egorecall.data.dataset import join_supervision
from egorecall.data.integrity import fingerprint_file, relative_file
from egorecall.data.scannetpp import ScanNetPPScene
from egorecall.data.scene_h5 import SceneH5
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
        required.update(scene["annotations"] for scene in json.load(stream))
    missing = required - files.keys()
    if missing:
        raise ValueError(f"Manifest omits required files: {sorted(missing)}.")
    return len(files)


def select_scenes(annotations: EgoRecallAnnotations, scene_ids: list[str] | None) -> tuple[str, ...]:
    """
    Restrict scene work to an explicit subset of a dataset's selected queries.

    Args:
        annotations: Annotation reader defining scene availability.
        scene_ids: Requested scene IDs, or None for all represented scenes.

    Returns:
        Unique scene IDs in the supplied order, or the reader's scene order.
    """
    if scene_ids is None:
        return annotations.scene_ids

    if not scene_ids or len(scene_ids) != len(set(scene_ids)):
        raise ValueError("Scene selection must be nonempty and contain no duplicates.")

    missing = set(scene_ids) - set(annotations.scene_ids)
    if missing:
        raise KeyError(f"Scenes are absent from the selected queries: {sorted(missing)}.")
    return tuple(scene_ids)


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

    # Verify the full package before restricting optional source/cache work to selected scenes.
    files_verified = verify_package(paths.dataset_root)
    with (paths.dataset_root / "manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    annotations = EgoRecallAnnotations(paths.dataset_root, split=manifest["selection"]["split"])
    selected = set(select_scenes(annotations, scene_ids))

    source_count = cache_count = decoded = 0
    object_count = 0

    for scene_id in annotations.scene_ids:
        scene_annotations = annotations.get_annotations(scene_id)
        object_count += len(scene_annotations["objects"])
        if scene_id not in selected:
            continue

        metadata = annotations.get_scene(scene_id)
        names = annotations.frame_names(scene_id)
        cameras = None
        fingerprints = None
        source_objects = None

        if check_source:
            # Validate source geometry and its object-ID/label join to the annotations.
            source = ScanNetPPScene(paths.scannetpp_root, scene_id)
            source_objects = source.objects()
            join_supervision(scene_id, source_objects, scene_annotations)

            # Match source camera names to the benchmark timeline.
            cameras = source.cameras(metadata["subsample_factor"])
            if cameras.frame_names != names:
                raise ValueError(f"{scene_id}: source timeline differs from the benchmark frame mapping.")

            if check_cache:
                fingerprints = source.cache_fingerprints()
            source_count += 1

        if check_cache:
            with SceneH5(paths.cache_root / f"{scene_id}.h5") as cached:
                cached.validate_compatibility(
                    scene_id,
                    names,
                    metadata["subsample_factor"],
                    metadata["source_fps"],
                    cameras=cameras,
                    source_files=fingerprints,
                    objects=source_objects,
                )

                # Cached geometry must join correctly even when raw-source checks are not requested.
                join_supervision(scene_id, cached.objects(), scene_annotations)

                # Decode the requested frame coverage only after compatibility checks pass.
                indices = range(len(names)) if decode_all else sorted({0, len(names) - 1})
                for index in indices:
                    cached.observation(index)
                    decoded += 1
            cache_count += 1

    # The reader checks table counts; full annotation loading also verifies object totals.
    counts = manifest["counts"]
    any_target_count = pc.sum(annotations.query_table["any_target"]).as_py()
    for name, actual in (("objects", object_count), ("any_target_queries", any_target_count)):
        if require_integer(counts[name], f"manifest/counts/{name}") != actual:
            raise ValueError(f"manifest/counts/{name}: declared {counts[name]}, found {actual}.")

    return CheckReport(files_verified, len(annotations), len(annotations.scene_ids), source_count, cache_count, decoded)
