"""
Read ScanNet++ camera records and object geometry from a user-supplied download.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from egorecall.data.integrity import FileFingerprint, fingerprint_file, relative_file
from egorecall.data.validation import require_integer, require_text
from scannetpp_common.annotations import load_annotation
from scannetpp_common.scene_release import ScannetppSceneRelease

DEPTH_SIZE = (256, 192)


@dataclass(frozen=True)
class CameraSequence:
    """
    Camera records ordered by canonical frame index. Poses map camera coordinates
    (x right, y down, z forward) into the mesh-aligned, Z-up world in metres.

    Args:
        frame_names: Source names corresponding to each canonical frame.
        camera_to_world: Float64 transforms with shape (N, 4, 4).
        intrinsics: Float64 pinhole matrices with shape (N, 3, 3) at image_size.
        timestamps: Source sensor timestamps in seconds, shape (N,).
        image_size: Native RGB (width, height).
    """

    frame_names: tuple[str, ...]
    camera_to_world: NDArray[np.float64]
    intrinsics: NDArray[np.float64]
    timestamps: NDArray[np.float64]
    image_size: tuple[int, int]


@dataclass(frozen=True)
class ObjectGeometry:
    """
    One source object's mesh-aligned geometry, in metres.

    Args:
        object_id: Positive ScanNet++ objectId, scoped to its scene.
        label: Source semantic label.
        centroid: Oriented-box centre, shape (3,).
        axes: Orthonormal box axes stored as rows, shape (3, 3).
        lengths: Full side lengths along those axes, shape (3,).
        minimum: Axis-aligned box minimum, shape (3,).
        maximum: Axis-aligned box maximum, shape (3,).
    """

    object_id: int
    label: str
    centroid: NDArray[np.float64]
    axes: NDArray[np.float64]
    lengths: NDArray[np.float64]
    minimum: NDArray[np.float64]
    maximum: NDArray[np.float64]

    def copy(self) -> ObjectGeometry:
        """
        Copy the record and its arrays so caller edits leave the original geometry unchanged.

        Returns:
            An independently owned geometry record.
        """
        return replace(
            self,
            centroid=self.centroid.copy(),
            axes=self.axes.copy(),
            lengths=self.lengths.copy(),
            minimum=self.minimum.copy(),
            maximum=self.maximum.copy(),
        )


def validate_object_geometry(obj: ObjectGeometry, context: str) -> None:
    """
    Check an object's box shapes, finite values, extents, and orthonormal axes.

    Args:
        obj: Object geometry from a source annotation or scene cache.
        context: Scene/object description used in errors.
    """
    for vector in (obj.centroid, obj.lengths, obj.minimum, obj.maximum):
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError(f"{context}: invalid bounding-box vectors.")

    if np.any(obj.lengths < 0) or np.any(obj.maximum < obj.minimum):
        raise ValueError(f"{context}: invalid bounding-box extents.")

    if (
        obj.axes.shape != (3, 3)
        or not np.isfinite(obj.axes).all()
        or not np.allclose(obj.axes @ obj.axes.T, np.eye(3), atol=1e-4)
    ):
        raise ValueError(f"{context}: box axes must be orthonormal rows.")


def object_geometry_sha256(objects: dict[int, ObjectGeometry]) -> str:
    """
    Fingerprint object IDs, labels, and box values in a deterministic representation.

    Args:
        objects: Validated source geometry keyed by object ID.

    Returns:
        SHA-256 of UTF-8 JSON with sorted IDs/keys, compact separators, and finite numbers.
    """
    records = [
        {
            "object_id": oid,
            "label": obj.label,
            "centroid": obj.centroid.tolist(),
            "axes": obj.axes.tolist(),
            "lengths": obj.lengths.tolist(),
            "minimum": obj.minimum.tolist(),
            "maximum": obj.maximum.tolist(),
        }
        for oid, obj in sorted(objects.items())
    ]
    encoded = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def source_frame_index(name: str) -> int:
    """
    Decode the source video-frame index from a ScanNet++ frame name.

    Args:
        name: A name such as frame_000010, without a file extension.

    Returns:
        Zero-based source index, distinct from the subsampled canonical index.
    """
    if not re.fullmatch(r"frame_[0-9]{6,}", name):
        raise ValueError(f"Invalid ScanNet++ frame name: {name!r}.")
    return int(name.removeprefix("frame_"))


def scale_intrinsics(
    intrinsics: NDArray[np.float64], source_size: tuple[int, int], target_size: tuple[int, int]
) -> NDArray[np.float64]:
    """
    Scale a pinhole matrix to a resized image or the corresponding sensor-depth grid.
    Pixel coordinates scale independently along x and y, without a half-pixel offset.

    Args:
        intrinsics: A matrix of shape (3, 3), or a batch with shape (N, 3, 3).
        source_size: Original (width, height).
        target_size: Requested (width, height).

    Returns:
        A new float64 array with scaled focal lengths and principal points.
    """
    for size in (source_size, target_size):
        for value in size:
            require_integer(value, "image dimension", minimum=1)

    scaled = np.array(intrinsics, dtype=np.float64, copy=True)
    if scaled.shape[-2:] != (3, 3) or scaled.ndim not in (2, 3):
        raise ValueError("Intrinsics must have shape (3, 3) or (N, 3, 3).")

    scaled[..., 0, :] *= target_size[0] / source_size[0]
    scaled[..., 1, :] *= target_size[1] / source_size[1]
    return scaled


def validate_cameras(cameras: CameraSequence) -> None:
    """
    Validate camera array shapes, finite values, and source ordering.
    Preserve the recorded transforms, including small floating-point deviations.

    Args:
        cameras: Canonically ordered camera records.
    """
    count = len(cameras.frame_names)
    indices = [source_frame_index(name) for name in cameras.frame_names]
    if not count or any(a >= b for a, b in zip(indices, indices[1:])):
        raise ValueError("Camera frame names must be nonempty, unique, and ordered by source index.")

    # All camera arrays must describe the same timeline with finite values.
    for array, shape in (
        (cameras.camera_to_world, (count, 4, 4)),
        (cameras.intrinsics, (count, 3, 3)),
        (cameras.timestamps, (count,)),
    ):
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f"Camera data must be finite with shape {shape}.")

    # Validate homogeneous poses and right-handed, orthonormal rotations.
    if not np.allclose(cameras.camera_to_world[:, 3, :], [0, 0, 0, 1], atol=1e-4):
        raise ValueError("Camera poses must be homogeneous camera-to-world transforms.")
    rotations = cameras.camera_to_world[:, :3, :3]
    if not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-3):
        raise ValueError("Camera rotations must be orthonormal.")
    if not np.allclose(np.linalg.det(rotations), 1.0, atol=1e-3):
        raise ValueError("Camera rotations must preserve handedness.")

    if not np.allclose(cameras.intrinsics[:, 2, :], [0, 0, 1]):
        raise ValueError("Intrinsics must be pinhole matrices.")
    if np.any(cameras.intrinsics[:, (0, 1), (0, 1)] <= 0):
        raise ValueError("Focal lengths must be positive.")

    if np.any(np.diff(cameras.timestamps) <= 0):
        raise ValueError("Camera timestamps must be strictly increasing.")

    for dimension in cameras.image_size:
        require_integer(dimension, "RGB dimension", minimum=1)


class ScanNetPPScene:
    """
    Access one scene in an original ScanNet++ download. Source objects include
    the complete annotation population; no EgoRecall visibility filter is applied.

    Args:
        root: Dataset root containing data and metadata, not the data subtree itself.
        scene_id: Scene to open.
    """

    def __init__(self, root: Path, scene_id: str) -> None:
        """
        Validate the dataset and scene directories and establish source paths.

        Args:
            root: Original ScanNet++ dataset root.
            scene_id: Scene directory name.
        """
        self.root = root.expanduser().resolve()
        if not (self.root / "data").is_dir() or not (self.root / "metadata").is_dir():
            raise FileNotFoundError(f"ScanNet++ root must contain data/ and metadata/: {self.root}.")

        self.scene_id = scene_id
        self.paths = ScannetppSceneRelease(scene_id, self.root / "data")
        if not self.paths.scene_root_dir.is_dir():
            raise FileNotFoundError(f"Missing ScanNet++ scene: {self.paths.scene_root_dir}.")

    def cameras(self, subsample_factor: int = 10) -> CameraSequence:
        """
        Build the timeline from sorted pose names sampled at the given stride.
        Use aligned_pose directly; world-to-camera transforms are its inverse.

        Args:
            subsample_factor: Pose-record stride, with 10 giving a nominal 6 FPS timeline.

        Returns:
            Validated camera records for the full sampled scene history.
        """
        require_integer(subsample_factor, "subsample_factor", minimum=1)
        with self.paths.iphone_pose_intrinsic_imu_path.open(encoding="utf-8") as stream:
            poses = json.load(stream)
        names = tuple(sorted(poses)[::subsample_factor])
        if not names:
            raise ValueError(f"{self.scene_id}: no camera records.")

        # EXIF dimensions describe the unrotated RGB grid used by the pinhole matrices.
        with self.paths.iphone_exif_path.open(encoding="utf-8") as stream:
            exif = json.load(stream)
        sizes = {(record["PixelXDimension"], record["PixelYDimension"]) for record in exif.values()}
        if len(sizes) != 1:
            raise ValueError(f"{self.scene_id}: EXIF must specify one consistent RGB resolution.")
        image_size = sizes.pop()

        # Assemble each camera field in the selected frame order.
        cameras = CameraSequence(
            frame_names=names,
            camera_to_world=np.array([poses[name]["aligned_pose"] for name in names], dtype=np.float64),
            intrinsics=np.array([poses[name]["intrinsic"] for name in names], dtype=np.float64),
            timestamps=np.array([poses[name]["timestamp"] for name in names], dtype=np.float64),
            image_size=image_size,
        )
        validate_cameras(cameras)
        return cameras

    def objects(self) -> dict[int, ObjectGeometry]:
        """
        Read every source object's label and oriented/axis-aligned boxes.

        Returns:
            Object geometry keyed by source objectId, with no visibility filtering.
        """
        objects: dict[int, ObjectGeometry] = {}
        for oid, record in load_annotation(self.paths.scan_anno_json_path).items():
            box = record["obb"]
            label = require_text(record["label"], f"{self.scene_id}/{oid}/label")
            obj = ObjectGeometry(
                object_id=oid,
                label=label,
                centroid=np.array(box["centroid"], dtype=np.float64),
                axes=np.array(box["normalizedAxes"], dtype=np.float64).reshape(3, 3),
                lengths=np.array(box["axesLengths"], dtype=np.float64),
                minimum=np.array(box["min"], dtype=np.float64),
                maximum=np.array(box["max"], dtype=np.float64),
            )
            validate_object_geometry(obj, f"{self.scene_id}/{oid}")
            objects[oid] = obj
        return objects

    def metadata_path(self, name: str) -> Path:
        """
        Locate an upstream metadata file, such as semantic_classes.txt.

        Args:
            name: Filename relative to the dataset's metadata directory.

        Returns:
            Path to an existing metadata file.
        """
        path = relative_file(self.root / "metadata", name)
        if not path.is_file():
            raise FileNotFoundError(f"Missing ScanNet++ metadata: {path}.")
        return path

    def cache_sources(self) -> dict[str, Path]:
        """
        Identify source files for observations, cameras, and object geometry in the scene cache.

        Returns:
            Paths keyed by filenames relative to the scene directory.
        """
        return {
            str(path.relative_to(self.paths.scene_root_dir)): path
            for path in (
                self.paths.iphone_pose_intrinsic_imu_path,
                self.paths.iphone_exif_path,
                self.paths.iphone_video_path,
                self.paths.iphone_video_mask_path,
                self.paths.iphone_depth_path,
                self.paths.scan_anno_json_path,
            )
        }

    def cache_fingerprints(self) -> dict[str, FileFingerprint]:
        """
        Hash scene-cache sources to detect changes before reusing prepared data.

        Returns:
            Source byte counts and SHA-256 digests, without machine-specific paths.
        """
        return {name: fingerprint_file(path) for name, path in self.cache_sources().items()}
