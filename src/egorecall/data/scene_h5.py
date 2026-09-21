"""
Read prepared observations, cameras, and source object geometry from one HDF5 cache per scene.

HDF5 layout (N is the number of canonical frames; M is the number of source objects):

  /                            Root attributes:
                                 format = "egorecall-observations"
                                 schema_version = 2
                                 scene_id: source scene identifier
                                 source_fps: nominal source frame rate, normally 60.0
                                 subsample_factor: sorted pose-record stride, normally 10
                                 rgb_resolution: (width, height) of the native RGB images
                                 depth_resolution: (256, 192)
                                 source_files: JSON mapping of scene-relative source paths
                                               to {"bytes": byte count, "sha256": hex digest}

  /frames/names                (N,) UTF-8 strings: source names such as frame_000010
  /frames/rgb_jpg              (N,) variable-length uint8: encoded RGB JPEG bytes
  /frames/depth_png            (N,) variable-length uint8: encoded uint16 depth PNG bytes
  /frames/mask_png             (N,) variable-length uint8: encoded grayscale mask PNG bytes
  /frames/rgb_jpg_sha256       (N,) S64: ASCII hexadecimal SHA-256 of each encoded RGB image
  /frames/depth_png_sha256     (N,) S64: ASCII hexadecimal SHA-256 of each encoded depth image
  /frames/mask_png_sha256      (N,) S64: ASCII hexadecimal SHA-256 of each encoded mask image

  /camera/aligned_pose         (N, 4, 4) float64: mesh-aligned camera-to-world transforms, metres
  /camera/intrinsic            (N, 3, 3) float64: pinhole matrices at native RGB resolution
  /camera/timestamp            (N,) float64: source sensor timestamps, seconds

  /objects                    Group attribute sha256: checksum of IDs, labels, and box values
  /objects/object_id          (M,) int64: all source objectId values, in increasing order
  /objects/label              (M,) UTF-8 strings: source semantic labels
  /objects/centroid           (M, 3) float64: oriented-box centres, metres
  /objects/axes               (M, 3, 3) float64: orthonormal box axes stored as rows
  /objects/lengths            (M, 3) float64: full box side lengths along each axis, metres
  /objects/minimum            (M, 3) float64: axis-aligned box minima, metres
  /objects/maximum            (M, 3) float64: axis-aligned box maxima, metres

Frame and camera datasets share the same zero-based canonical index; /frames/names
maps that index to the source frame name. Each image entry stores compressed bytes.
Decoded RGB has shape (height, width, 3) in RGB order, masks have shape (height, width),
and depth has shape (192, 256) in uint16 millimetres, with zero indicating invalid depth.
Depth intrinsics are computed by scaling the RGB intrinsics to the sensor-depth grid.
Camera axes are x-right, y-down, z-forward in a mesh-aligned world with Z pointing up.
Object rows are aligned by /objects/object_id and retain the complete source population.
The objects checksum uses the canonical JSON representation in object_geometry_sha256().
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import cast

import h5py
import numpy as np
from numpy.typing import NDArray

from egorecall.data.integrity import FileFingerprint
from egorecall.data.media import decode_image
from egorecall.data.scannetpp import (
    DEPTH_SIZE,
    CameraSequence,
    ObjectGeometry,
    object_geometry_sha256,
    scale_intrinsics,
    validate_cameras,
    validate_object_geometry,
)
from egorecall.data.validation import require_integer, require_text

CACHE_VERSION = 2
IMAGE_DATASETS = {"rgb": "rgb_jpg", "depth": "depth_png", "mask": "mask_png"}
OBJECT_ARRAY_SHAPES = {"centroid": (3,), "axes": (3, 3), "lengths": (3,), "minimum": (3,), "maximum": (3,)}


@dataclass(frozen=True)
class Observation:
    """
    One RGB-D observation and its source camera, with no object annotations.

    Args:
        frame_idx: Zero-based canonical index.
        frame_name: Corresponding source frame name.
        timestamp: Source timestamp in seconds.
        rgb: uint8 RGB image, shape (H, W, 3), in native pixel orientation.
        depth: uint16 sensor depth in millimetres, shape (192, 256); zero is invalid.
        mask: uint8 source anonymization mask at RGB resolution, without reinterpretation.
        camera_to_world: Mesh-aligned 4x4 camera-to-world transform in metres.
        rgb_intrinsics: Pinhole intrinsics for the native RGB grid.
        depth_intrinsics: The same pinhole model scaled to the sensor-depth grid.
    """

    frame_idx: int
    frame_name: str
    timestamp: float
    rgb: NDArray[np.uint8]
    depth: NDArray[np.uint16]
    mask: NDArray[np.uint8]
    camera_to_world: NDArray[np.float64]
    rgb_intrinsics: NDArray[np.float64]
    depth_intrinsics: NDArray[np.float64]


class SceneH5:
    """
    Open a scene cache and validate its structure and object geometry. Use as a
    context manager to close the HDF5 handle. Encoded-frame checksums are verified when read.

    Args:
        path: Prepared scene HDF5 file containing observations and source geometry.
    """

    def __init__(self, path: Path) -> None:
        """
        Open a completed cache and validate metadata, cameras, frame datasets, and objects.

        Args:
            path: Scene-cache path.
        """
        self.path = path
        self._file = h5py.File(path, "r")
        try:
            self._validate_structure()
        except BaseException:
            self._file.close()
            raise

    def _validate_structure(self) -> None:
        """
        Require the scene-cache schema and complete, consistent frame and object arrays.
        """
        attrs = self._file.attrs
        if attrs["format"] != "egorecall-observations" or attrs["schema_version"] != CACHE_VERSION:
            raise ValueError(
                f"{self.path}: expected scene-cache schema {CACHE_VERSION} with object geometry; "
                "recreate this cache using the current preparation command in a new cache directory."
            )

        for name in ("schema_version", "subsample_factor"):
            if isinstance(attrs[name], (bool, np.bool_)) or not isinstance(attrs[name], (int, np.integer)):
                raise ValueError(f"Cache {name} must be an integer.")

        self.scene_id = attrs["scene_id"]
        if not isinstance(self.scene_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", self.scene_id):
            raise ValueError("Invalid cache scene_id.")

        if isinstance(attrs["source_fps"], (bool, np.bool_)) or not isinstance(
            attrs["source_fps"], (int, float, np.integer, np.floating)
        ):
            raise ValueError("Cache source_fps must be numeric.")
        self.source_fps = float(attrs["source_fps"])
        if not np.isfinite(self.source_fps) or self.source_fps <= 0:
            raise ValueError("Cache source_fps must be finite and positive.")

        self.subsample_factor = int(attrs["subsample_factor"])
        require_integer(self.subsample_factor, "cache subsample_factor", minimum=1)

        for name in ("rgb_resolution", "depth_resolution"):
            dimensions = np.asarray(attrs[name])
            if dimensions.shape != (2,) or dimensions.dtype.kind not in "iu" or np.any(dimensions <= 0):
                raise ValueError(f"Invalid cache {name}.")
        if tuple(attrs["depth_resolution"]) != DEPTH_SIZE:
            raise ValueError("Invalid cache depth_resolution.")
        self.image_size = (int(attrs["rgb_resolution"][0]), int(attrs["rgb_resolution"][1]))

        self.frame_names = tuple(self._file["frames/names"].asstr()[:])

        # Camera arrays are small enough to load once; images remain on demand.
        self._cameras = CameraSequence(
            self.frame_names,
            self._file["camera/aligned_pose"][:],
            self._file["camera/intrinsic"][:],
            self._file["camera/timestamp"][:],
            self.image_size,
        )
        validate_cameras(self._cameras)
        for name in ("aligned_pose", "intrinsic", "timestamp"):
            if self._file[f"camera/{name}"].dtype != np.float64:
                raise ValueError(f"camera/{name}: expected float64 values.")

        # Encoded images and their checksums must cover the full camera timeline.
        for name in IMAGE_DATASETS.values():
            encoded = self._file[f"frames/{name}"]
            hashes = self._file[f"frames/{name}_sha256"]
            if encoded.shape != (len(self.frame_names),) or hashes.shape != encoded.shape:
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
        self._source_files = cast(dict[str, FileFingerprint], fingerprints)
        if "scans/segments_anno.json" not in self._source_files:
            raise ValueError("Cache source_files must include scans/segments_anno.json.")

        # Load the small geometry table once and verify it independently of raw files.
        self._objects = self._load_objects()

    def _load_objects(self) -> dict[int, ObjectGeometry]:
        """
        Validate object columns, construct typed geometry, and verify the stored checksum.

        Returns:
            Full source-object population owned by this cache reader.
        """
        group = self._file["objects"]
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
        names = labels.asstr()[:]

        arrays: dict[str, NDArray[np.float64]] = {}
        for name, shape in OBJECT_ARRAY_SHAPES.items():
            column = group[name]
            if column.shape != (count, *shape) or column.dtype != np.dtype(np.float64):
                raise ValueError(f"objects/{name} must be a float64 array with shape {(count, *shape)}.")
            arrays[name] = column[:]

        objects: dict[int, ObjectGeometry] = {}
        for position, value in enumerate(object_ids):
            oid = int(value)
            context = f"{self.scene_id}/{oid}"
            label = require_text(names[position], f"{context}/label")
            obj = ObjectGeometry(
                oid,
                label,
                arrays["centroid"][position],
                arrays["axes"][position],
                arrays["lengths"][position],
                arrays["minimum"][position],
                arrays["maximum"][position],
            )
            validate_object_geometry(obj, context)
            objects[oid] = obj

        self._objects_sha256 = object_geometry_sha256(objects)
        if self._objects_sha256 != group.attrs["sha256"]:
            raise ValueError(f"{self.path}: object geometry checksum mismatch.")
        return objects

    def objects(self) -> dict[int, ObjectGeometry]:
        """
        Read all cached source objects with independently owned geometry arrays.

        Returns:
            Object IDs, labels, and boxes, including objects outside the EgoRecall filtered population.
        """
        return {oid: obj.copy() for oid, obj in self._objects.items()}

    def validate_compatibility(
        self,
        scene_id: str,
        frame_names: tuple[str, ...],
        subsample_factor: int,
        source_fps: float,
        *,
        cameras: CameraSequence | None = None,
        source_files: dict[str, FileFingerprint] | None = None,
        objects: dict[int, ObjectGeometry] | None = None,
    ) -> None:
        """
        Require matching scene identity and timeline before using or reusing a cache.
        Source cameras, objects, and fingerprints can additionally verify the original download.

        Args:
            scene_id: Expected source scene.
            frame_names: Complete canonical timeline, including later query times.
            subsample_factor: Expected pose-record stride.
            source_fps: Nominal source frame rate.
            cameras: Source camera records, when checking against the raw download.
            source_files: Current source fingerprints, when checking cache reuse.
            objects: Source object geometry, when checking against the raw download.
        """
        if (
            self.scene_id != scene_id
            or self.frame_names != frame_names
            or self.subsample_factor != subsample_factor
            or self.source_fps != source_fps
        ):
            raise ValueError(f"{self.path}: cache scene or timeline does not match the requested data.")

        if source_files is not None and self._source_files != source_files:
            raise ValueError(
                f"{self.path}: source files changed; use a new cache directory or remove this stale cache."
            )

        if cameras is not None:
            if cameras.image_size != self.image_size or not all(
                np.array_equal(expected, actual)
                for expected, actual in (
                    (cameras.camera_to_world, self._cameras.camera_to_world),
                    (cameras.intrinsics, self._cameras.intrinsics),
                    (cameras.timestamps, self._cameras.timestamps),
                )
            ):
                raise ValueError(f"{self.path}: cached camera values differ from the source.")

        if objects is not None and object_geometry_sha256(objects) != self._objects_sha256:
            raise ValueError(f"{self.path}: cached object geometry differs from the source.")

    def encoded_image(self, frame_idx: int, kind: str) -> bytes:
        """
        Read and checksum an encoded frame without decoding its pixels.

        Args:
            frame_idx: Zero-based canonical index within the cache timeline.
            kind: One of rgb, depth, or mask.

        Returns:
            JPEG or PNG bytes suitable for independent image decoders.
        """
        self._require_frame(frame_idx)
        name = IMAGE_DATASETS[kind]
        payload = self._file[f"frames/{name}"][frame_idx].tobytes()
        expected = self._file[f"frames/{name}_sha256"][frame_idx].decode("ascii")
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"{self.path}: checksum mismatch at {kind} frame {frame_idx}.")
        return payload

    def observation(self, frame_idx: int) -> Observation:
        """
        Decode one frame with independently owned image and camera arrays.

        Args:
            frame_idx: Zero-based canonical index within the cache timeline.

        Returns:
            Native RGB, sensor depth, anonymization mask, pose, and scaled intrinsics.
        """
        self._require_frame(frame_idx)

        # Decode the image payloads at their native RGB and sensor-depth resolutions.
        rgb = cast(NDArray[np.uint8], decode_image(self.encoded_image(frame_idx, "rgb"), "rgb", self.image_size))
        depth = cast(NDArray[np.uint16], decode_image(self.encoded_image(frame_idx, "depth"), "depth", DEPTH_SIZE))
        mask = cast(NDArray[np.uint8], decode_image(self.encoded_image(frame_idx, "mask"), "mask", self.image_size))

        # Copy camera values and scale intrinsics for the returned depth grid.
        intrinsic = self._cameras.intrinsics[frame_idx].copy()
        return Observation(
            frame_idx,
            self.frame_names[frame_idx],
            float(self._cameras.timestamps[frame_idx]),
            rgb,
            depth,
            mask,
            self._cameras.camera_to_world[frame_idx].copy(),
            intrinsic,
            scale_intrinsics(intrinsic, self.image_size, DEPTH_SIZE),
        )

    def _require_frame(self, frame_idx: int) -> None:
        """
        Reject negative, noninteger, and out-of-range canonical indices.

        Args:
            frame_idx: Requested canonical index.
        """
        require_integer(frame_idx, "frame_idx")
        if frame_idx >= len(self.frame_names):
            raise IndexError(f"Frame {frame_idx} is outside the {len(self.frame_names)}-frame cache.")

    def close(self) -> None:
        """
        Close the HDF5 handle. Subsequent image reads require reopening the cache.
        """
        self._file.close()

    def __enter__(self) -> SceneH5:
        """
        Enter a context that owns this cache handle.

        Returns:
            This open reader.
        """
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        """
        Close the handle on normal exit or failure.

        Args:
            exc_type: Active exception type, if any.
            exc: Active exception, if any.
            traceback: Active exception traceback, if any.
        """
        self.close()
