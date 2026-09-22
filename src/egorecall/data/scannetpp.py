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
from egorecall.data.validation import require_integer
from scannetpp_common.annotations import load_annotation
from scannetpp_common.scene_release import ScannetppSceneRelease

DEPTH_SIZE = (256, 192)


@dataclass(frozen=True)
class CameraSequence:
    """
    Camera records ordered by sampled frame index. Poses map camera coordinates
    (x right, y down, z forward) into the mesh-aligned, Z-up world in metres.

    Args:
        frame_names: Source names corresponding to each sampled frame.
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


def object_geometry_sha256(objects_by_id: dict[int, ObjectGeometry]) -> str:
    """
    Fingerprint object IDs, labels, and box values in a deterministic representation.

    Args:
        objects_by_id: Validated source geometry keyed by object ID.

    Returns:
        SHA-256 of UTF-8 JSON with sorted IDs/keys, compact separators, and finite numbers.
    """
    records = [
        {
            "object_id": oid,
            "label": object_geometry.label,
            "centroid": object_geometry.centroid.tolist(),
            "axes": object_geometry.axes.tolist(),
            "lengths": object_geometry.lengths.tolist(),
            "minimum": object_geometry.minimum.tolist(),
            "maximum": object_geometry.maximum.tolist(),
        }
        for oid, object_geometry in sorted(objects_by_id.items())
    ]
    encoded = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def source_frame_index(name: str) -> int:
    """
    Decode the source video-frame index from a ScanNet++ frame name.

    Args:
        name: A name such as frame_000010, without a file extension.

    Returns:
        The number in the filename: frame_000010 returns 10, even when it is only the second sampled image.
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


class ScanNetPPScene:
    """
    Access one scene in an original ScanNet++ download. Source objects include
    the complete annotation population; no EgoRecall visibility filter is applied.

    Args:
        root: Dataset root containing data, not the data subtree itself.
        scene_id: Scene to open.
    """

    def __init__(self, root: Path, scene_id: str) -> None:
        """
        Locate source files under data/<scene_id>; files are opened by the accessors.

        Args:
            root: Original ScanNet++ dataset root.
            scene_id: Scene directory name.
        """
        self.root = root.expanduser().resolve()

        self.scene_id = scene_id
        self.paths = ScannetppSceneRelease(scene_id, self.root / "data")

    def cameras(self, subsample_factor: int = 10, *, frame_names: tuple[str, ...] | None = None) -> CameraSequence:
        """
        Read cameras for the supplied frame names, or sample sorted pose names at the given stride.
        Use aligned_pose directly; world-to-camera transforms are its inverse.

        Args:
            subsample_factor: Pose-record stride, with 10 giving a nominal 6 FPS timeline.
            frame_names: Source frames to read in order, or None to select them using the stride.

        Returns:
            Camera poses, intrinsics, and timestamps in the sampled frame order.
        """
        require_integer(subsample_factor, "subsample_factor", minimum=1)
        with self.paths.iphone_pose_intrinsic_imu_path.open(encoding="utf-8") as stream:
            pose_records = json.load(stream)
        if frame_names is None:
            frame_names = tuple(sorted(pose_records)[::subsample_factor])

        # EXIF dimensions describe the unrotated RGB grid used by the pinhole matrices.
        with self.paths.iphone_exif_path.open(encoding="utf-8") as stream:
            exif_records = json.load(stream)
        first_exif = next(iter(exif_records.values()))
        image_size = (first_exif["PixelXDimension"], first_exif["PixelYDimension"])

        # Assemble each camera field in the selected frame order.
        camera_sequence = CameraSequence(
            frame_names=frame_names,
            camera_to_world=np.array([pose_records[name]["aligned_pose"] for name in frame_names], dtype=np.float64),
            intrinsics=np.array([pose_records[name]["intrinsic"] for name in frame_names], dtype=np.float64),
            timestamps=np.array([pose_records[name]["timestamp"] for name in frame_names], dtype=np.float64),
            image_size=image_size,
        )
        return camera_sequence

    def objects(self) -> dict[int, ObjectGeometry]:
        """
        Read every source object's label and oriented/axis-aligned boxes.

        Returns:
            Object geometry keyed by source objectId, with no visibility filtering.
        """
        objects_by_id: dict[int, ObjectGeometry] = {}
        for oid, object_record in load_annotation(self.paths.scan_anno_json_path).items():
            box = object_record["obb"]
            object_geometry = ObjectGeometry(
                object_id=oid,
                label=object_record["label"],
                centroid=np.array(box["centroid"], dtype=np.float64),
                axes=np.array(box["normalizedAxes"], dtype=np.float64).reshape(3, 3),
                lengths=np.array(box["axesLengths"], dtype=np.float64),
                minimum=np.array(box["min"], dtype=np.float64),
                maximum=np.array(box["max"], dtype=np.float64),
            )
            objects_by_id[oid] = object_geometry
        return objects_by_id

    def metadata_path(self, name: str) -> Path:
        """
        Locate an upstream metadata file, such as semantic_classes.txt.

        Args:
            name: Filename relative to the dataset's metadata directory.

        Returns:
            Path to the named metadata file; opening it reports a missing file.
        """
        return relative_file(self.root / "metadata", name)

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
        Hash the source files so the checker can detect changes after preparing a cache.

        Returns:
            Source byte counts and SHA-256 digests, without machine-specific paths.
        """
        return {name: fingerprint_file(path) for name, path in self.cache_sources().items()}
