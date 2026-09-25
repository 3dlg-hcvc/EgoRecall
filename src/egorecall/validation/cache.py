"""
Validate prepared H5 caches explicitly, including stored checksums and image encodings.
"""

import hashlib
import json
import re
from io import BytesIO
from pathlib import Path

import h5py
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from egorecall.arguments import require_integer, require_text
from egorecall.data.images import validate_image
from egorecall.data.scannetpp import SCENE_ID_PATTERN
from egorecall.data.scene_h5 import (
    CACHE_VERSION,
    DEPTH_SIZE,
    IMAGE_DATASETS,
    OBJECT_ARRAY_SHAPES,
    object_geometry_sha256,
)
from egorecall.geometry import CameraSequence, ObjectGeometry
from egorecall.integrity import FileFingerprint
from egorecall.validation.sources import validate_cameras, validate_object_geometry


def validate_scene_cache(path: Path, *, decode_all: bool = False) -> int:
    """
    Check H5 metadata and every encoded image's checksum and header. Decode the
    first and last frames as an additional image-codec check, or every frame when requested.

    Args:
        path: Prepared scene H5 file.
        decode_all: Decode all RGB, depth, and mask images rather than just the endpoints.

    Returns:
        Number of frames for which all three images were decoded.
    """
    with h5py.File(path, "r") as h5_file:
        validate_cache_structure(h5_file)
        num_frames = len(h5_file["frames/names"])
        decoded_indices = range(num_frames) if decode_all else {0, num_frames - 1}
        rgb_size = tuple(h5_file.attrs["rgb_resolution"])
        for frame_idx in range(num_frames):
            for image_kind, dataset_name in IMAGE_DATASETS.items():
                payload = h5_file[f"frames/{dataset_name}"][frame_idx].tobytes()
                expected = h5_file[f"frames/{dataset_name}_sha256"][frame_idx].decode("ascii")
                if hashlib.sha256(payload).hexdigest() != expected:
                    raise ValueError(f"{path}: checksum mismatch at {image_kind} frame {frame_idx}.")
                image_size = DEPTH_SIZE if image_kind == "depth" else rgb_size
                validate_image(payload, image_kind, image_size)
                if frame_idx in decoded_indices:
                    with Image.open(BytesIO(payload)) as image:
                        image.load()
        return len(decoded_indices)


def validate_cache_structure(h5_file: h5py.File) -> None:
    """
    Check H5 attributes, array sizes, camera values, source fingerprints, and
    object geometry before any frame payloads are read.

    Args:
        h5_file: Open scene cache to inspect.
    """
    attrs = h5_file.attrs
    if attrs["format"] != "egorecall-observations" or attrs["schema_version"] != CACHE_VERSION:
        raise ValueError(
            f"{h5_file.filename}: unsupported scene-cache format or schema_version; expected schema {CACHE_VERSION}. "
            "Prepare the scene again in a new cache directory."
        )

    for name in ("schema_version", "subsample_factor"):
        if isinstance(attrs[name], (bool, np.bool_)) or not isinstance(attrs[name], (int, np.integer)):
            raise ValueError(f"Cache {name} must be an integer.")

    scene_id = attrs["scene_id"]
    if not isinstance(scene_id, str) or not re.fullmatch(SCENE_ID_PATTERN, scene_id):
        raise ValueError("Invalid cache scene_id.")

    if isinstance(attrs["source_fps"], (bool, np.bool_)) or not isinstance(
        attrs["source_fps"], (int, float, np.integer, np.floating)
    ):
        raise ValueError("Cache source_fps must be numeric.")
    source_fps = float(attrs["source_fps"])
    if not np.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("Cache source_fps must be finite and positive.")

    subsample_factor = int(attrs["subsample_factor"])
    require_integer(subsample_factor, "cache subsample_factor", minimum=1)

    for name in ("rgb_resolution", "depth_resolution"):
        dimensions = np.asarray(attrs[name])
        if dimensions.shape != (2,) or dimensions.dtype.kind not in "iu" or np.any(dimensions <= 0):
            raise ValueError(f"Invalid cache {name}.")
    if tuple(attrs["depth_resolution"]) != DEPTH_SIZE:
        raise ValueError("Invalid cache depth_resolution.")
    image_size = (int(attrs["rgb_resolution"][0]), int(attrs["rgb_resolution"][1]))

    frame_names = tuple(h5_file["frames/names"].asstr()[:])

    # Check camera arrays before using their frame count to check the image datasets.
    camera_sequence = CameraSequence(
        frame_names,
        h5_file["camera/aligned_pose"][:],
        h5_file["camera/intrinsic"][:],
        h5_file["camera/timestamp"][:],
        image_size,
    )
    validate_cameras(camera_sequence)
    for name in ("aligned_pose", "intrinsic", "timestamp"):
        if h5_file[f"camera/{name}"].dtype != np.float64:
            raise ValueError(f"camera/{name}: expected float64 values.")

    # Encoded images and their checksums must cover the full camera timeline.
    for name in IMAGE_DATASETS.values():
        encoded = h5_file[f"frames/{name}"]
        hashes = h5_file[f"frames/{name}_sha256"]
        if encoded.shape != (len(frame_names),) or hashes.shape != encoded.shape:
            raise ValueError(f"{name}: image count does not match the camera timeline.")
        if h5py.check_vlen_dtype(encoded.dtype) != np.dtype(np.uint8) or hashes.dtype != np.dtype("S64"):
            raise ValueError(f"{name}: incompatible encoded-frame or checksum type.")

    # Validate the source fingerprints used for cache compatibility checks.
    fingerprints = json.loads(attrs["source_files"])
    if not isinstance(fingerprints, dict) or not fingerprints:
        raise ValueError("Cache source_files must contain source fingerprints.")
    for name, value in fingerprints.items():
        require_integer(value["bytes"], f"{name}/bytes", minimum=1)
        if not isinstance(value["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", value["sha256"]):
            raise ValueError(f"{name}: invalid source SHA-256 digest.")

    # Object IDs, labels, box columns, and their checksum must agree.
    _validate_cached_objects(h5_file, scene_id)


def _validate_cached_objects(h5_file: h5py.File, scene_id: str) -> None:
    """
    Validate object columns, construct typed geometry, and verify the stored checksum.

    Args:
        h5_file: Open scene cache.
        scene_id: Scene ID used in error messages.
    """
    group = h5_file["objects"]
    ids = group["object_id"]
    if ids.ndim != 1 or ids.dtype != np.dtype(np.int64):
        raise ValueError("objects/object_id must be a one-dimensional int64 array.")
    object_ids = ids[:]
    if np.any(object_ids <= 0) or np.any(object_ids[1:] <= object_ids[:-1]):
        raise ValueError("Cached object IDs must be positive, unique, and increasing.")
    count = len(object_ids)

    labels = group["label"]
    string_type = h5py.check_string_dtype(labels.dtype)
    if labels.shape != (count,) or string_type is None or string_type.encoding != "utf-8":
        raise ValueError("objects/label must contain one UTF-8 label per object.")
    object_labels = labels.asstr()[:]

    arrays: dict[str, NDArray[np.float64]] = {}
    for name, shape in OBJECT_ARRAY_SHAPES.items():
        column = group[name]
        if column.shape != (count, *shape) or column.dtype != np.dtype(np.float64):
            raise ValueError(f"objects/{name} must be a float64 array with shape {(count, *shape)}.")
        arrays[name] = column[:]

    objects_by_id: dict[int, ObjectGeometry] = {}
    for position, value in enumerate(object_ids):
        oid = int(value)
        context = f"{scene_id}/{oid}"
        label = require_text(object_labels[position], f"{context}/label")
        object_geometry = ObjectGeometry(
            oid,
            label,
            arrays["centroid"][position],
            arrays["axes"][position],
            arrays["lengths"][position],
            arrays["minimum"][position],
            arrays["maximum"][position],
        )
        validate_object_geometry(object_geometry, context)
        objects_by_id[oid] = object_geometry

    objects_sha256 = object_geometry_sha256(objects_by_id)
    if objects_sha256 != group.attrs["sha256"]:
        raise ValueError(f"{h5_file.filename}: object geometry checksum mismatch.")


def validate_cache_compatibility(
    h5_file: h5py.File,
    scene_id: str,
    frame_names: tuple[str, ...],
    subsample_factor: int,
    source_fps: float,
    *,
    cameras: CameraSequence | None = None,
    source_files: dict[str, FileFingerprint] | None = None,
    objects_by_id: dict[int, ObjectGeometry] | None = None,
) -> None:
    """
    Compare a cache with annotation settings and, when supplied, the source files
    used to prepare it. This detects stale caches and mixing data from different scenes.

    Args:
        h5_file: Open scene cache whose structure has already been checked.
        scene_id: Expected source scene.
        frame_names: Complete sampled frame sequence, including later query times.
        subsample_factor: Expected pose-record stride.
        source_fps: Nominal source frame rate.
        cameras: Source camera records, when checking against the raw download.
        source_files: Current source fingerprints, when checking cache reuse.
        objects_by_id: Source object geometry, when checking against the raw download.
    """
    # Report each differing setting on its own; a different stride also changes the frame names.
    cached_scene_id = h5_file.attrs["scene_id"]
    if cached_scene_id != scene_id:
        raise ValueError(f"{h5_file.filename}: cache holds scene {cached_scene_id!r}, expected {scene_id!r}.")
    cached_stride = h5_file.attrs["subsample_factor"]
    if cached_stride != subsample_factor:
        raise ValueError(f"{h5_file.filename}: cache sampling stride is {cached_stride}, expected {subsample_factor}.")
    cached_fps = h5_file.attrs["source_fps"]
    if cached_fps != source_fps:
        raise ValueError(f"{h5_file.filename}: cache source_fps is {cached_fps}, expected {source_fps}.")
    if tuple(h5_file["frames/names"].asstr()[:]) != frame_names:
        raise ValueError(f"{h5_file.filename}: cached frame names differ from the expected timeline.")

    if source_files is not None and json.loads(h5_file.attrs["source_files"]) != source_files:
        raise ValueError(
            f"{h5_file.filename}: source files changed; use a new cache directory or remove this stale cache."
        )

    if cameras is not None:
        if cameras.image_size != tuple(h5_file.attrs["rgb_resolution"]) or not all(
            np.array_equal(expected, actual)
            for expected, actual in (
                (cameras.camera_to_world, h5_file["camera/aligned_pose"][:]),
                (cameras.intrinsics, h5_file["camera/intrinsic"][:]),
                (cameras.timestamps, h5_file["camera/timestamp"][:]),
            )
        ):
            raise ValueError(f"{h5_file.filename}: cached camera values differ from the source.")

    if objects_by_id is not None and object_geometry_sha256(objects_by_id) != h5_file["objects"].attrs["sha256"]:
        raise ValueError(f"{h5_file.filename}: cached object geometry differs from the source.")
