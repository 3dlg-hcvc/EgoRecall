"""
Prepare canonical observations and source object geometry in a local HDF5 scene cache.
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np

from egorecall.data.images import encode_depth, validate_image
from egorecall.data.scannetpp import SOURCE_FPS, ScanNetPPScene, source_frame_index
from egorecall.data.scene_h5 import (
    CACHE_VERSION,
    DEPTH_SIZE,
    IMAGE_DATASETS,
    OBJECT_ARRAY_SHAPES,
    object_geometry_sha256,
)
from egorecall.geometry import ObjectGeometry
from egorecall.preparation.depth import iter_depth_frames
from egorecall.preparation.video import extract_video_frames


def prepare_scene(
    source_scene: ScanNetPPScene,
    cache_root: Path,
    *,
    subsample_factor: int = 10,
    source_fps: float = SOURCE_FPS,
    frame_names: tuple[str, ...] | None = None,
    ffmpeg: str = "ffmpeg",
) -> Path:
    """
    Prepare all sampled images and source objects. Existing cache files are left in
    place; use egorecall-check to verify them. A temporary file is published atomically; interrupted work
    never replaces a completed cache. Source files are read from their original paths.

    Args:
        source_scene: Scene in the original ScanNet++ download.
        cache_root: Separate writable directory for one H5 file per scene.
        subsample_factor: Sorted pose-record stride; the benchmark uses 10.
        source_fps: Nominal source frame rate, stored as metadata only.
        frame_names: Source frames selected by the annotation package, or None to use the sampling stride.
        ffmpeg: FFmpeg executable name or path.

    Returns:
        Path to the newly prepared or existing scene cache.
    """
    cache_root = cache_root.expanduser().resolve()
    if cache_root.is_relative_to(source_scene.root):
        raise ValueError("cache_root must be outside the ScanNet++ source directory.")

    if not np.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("source_fps must be finite and positive.")

    output_path = cache_root / f"{source_scene.scene_id}.h5"
    if output_path.exists():
        return output_path

    # Read source cameras and boxes once; the checker compares them with annotations.
    camera_sequence = source_scene.cameras(subsample_factor, frame_names=frame_names)
    frame_names = camera_sequence.frame_names
    source_objects = source_scene.objects()
    fingerprints = source_scene.cache_fingerprints()

    # Extract loose images in system temporary storage; build the H5 beside its
    # destination so publication remains atomic even when the cache is on another filesystem.
    cache_root.mkdir(parents=True, exist_ok=True)
    with (
        tempfile.TemporaryDirectory(prefix=f"egorecall-{source_scene.scene_id}-") as temporary,
        tempfile.TemporaryDirectory(prefix=f".{source_scene.scene_id}-", dir=cache_root) as staging,
    ):
        work_dir = Path(temporary)
        rgb_files = extract_video_frames(source_scene.iphone_video_path, frame_names, work_dir / "rgb", ffmpeg=ffmpeg)
        mask_files = extract_video_frames(
            source_scene.iphone_video_mask_path, frame_names, work_dir / "mask", masks=True, ffmpeg=ffmpeg
        )

        # Write encoded frames one at a time, retaining native depth values and RGB orientation.
        temporary_h5 = Path(staging) / "scene.h5"
        with h5py.File(temporary_h5, "w") as h5_file:
            h5_file.attrs.update(
                format="egorecall-observations",
                schema_version=CACHE_VERSION,
                scene_id=source_scene.scene_id,
                source_fps=source_fps,
                subsample_factor=subsample_factor,
                rgb_resolution=camera_sequence.image_size,
                depth_resolution=DEPTH_SIZE,
                source_files=json.dumps(fingerprints, sort_keys=True),
            )

            # Allocate the timeline and encoded-image datasets before writing payloads.
            h5_file.create_dataset("frames/names", data=frame_names, dtype=h5py.string_dtype("utf-8"))
            for name in IMAGE_DATASETS.values():
                h5_file.create_dataset(f"frames/{name}", (len(frame_names),), dtype=h5py.vlen_dtype(np.uint8))
                h5_file.create_dataset(f"frames/{name}_sha256", (len(frame_names),), dtype="S64")

            h5_file.create_dataset("camera/aligned_pose", data=camera_sequence.camera_to_world)
            h5_file.create_dataset("camera/intrinsic", data=camera_sequence.intrinsics)
            h5_file.create_dataset("camera/timestamp", data=camera_sequence.timestamps)

            # Store source geometry locally for supervision without reopening the raw download.
            _write_objects(h5_file, source_objects)

            # Pack RGB and mask images in sampled frame order.
            for frame_idx, (rgb, mask) in enumerate(zip(rgb_files, mask_files, strict=True)):
                for kind, image_path in (("rgb", rgb), ("mask", mask)):
                    payload = image_path.read_bytes()
                    validate_image(payload, kind, camera_sequence.image_size)
                    _write_image(h5_file, frame_idx, kind, payload)

            # Depth source indices are independent of sampled frame indices and can have gaps.
            position_by_source = {source_frame_index(name): position for position, name in enumerate(frame_names)}
            remaining = set(position_by_source)
            for source_idx, depth in iter_depth_frames(
                source_scene.iphone_depth_path, selected=set(position_by_source)
            ):
                _write_image(h5_file, position_by_source[source_idx], "depth", encode_depth(depth))
                remaining.remove(source_idx)

            if remaining:
                raise ValueError(f"{source_scene.scene_id}: depth is missing source frames {sorted(remaining)[:5]}.")

        # Linking the completed temporary file cannot overwrite an existing result.
        os.link(temporary_h5, output_path)
    return output_path


def _write_objects(h5_file: h5py.File, source_objects: dict[int, ObjectGeometry]) -> None:
    """
    Store all source object IDs, labels, and boxes with a checksum of their values.

    Args:
        h5_file: Writable scene H5 file.
        source_objects: All source objects keyed by objectId.
    """
    group = h5_file.create_group("objects")
    ids = sorted(source_objects)
    group.create_dataset("object_id", data=np.array(ids, dtype=np.int64))
    group.create_dataset("label", data=[source_objects[oid].label for oid in ids], dtype=h5py.string_dtype("utf-8"))

    # Every geometry column follows the same object-ID order, including empty populations.
    for name, shape in OBJECT_ARRAY_SHAPES.items():
        values = np.array([getattr(source_objects[oid], name) for oid in ids], dtype=np.float64).reshape(
            len(ids), *shape
        )
        group.create_dataset(name, data=values)
    group.attrs["sha256"] = object_geometry_sha256(source_objects)


def _write_image(h5_file: h5py.File, frame_idx: int, kind: str, payload: bytes) -> None:
    """
    Store image bytes and a checksum for the standalone cache checker.

    Args:
        h5_file: Writable scene H5 file.
        frame_idx: Zero-based position in the sampled frame sequence.
        kind: One of rgb, depth, or mask.
        payload: Complete encoded image.
    """
    name = IMAGE_DATASETS[kind]
    h5_file[f"frames/{name}"][frame_idx] = np.frombuffer(payload, dtype=np.uint8)
    h5_file[f"frames/{name}_sha256"][frame_idx] = hashlib.sha256(payload).hexdigest().encode("ascii")
