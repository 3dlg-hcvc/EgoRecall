"""
Read prepared observations, cameras, and source object geometry from one HDF5 cache per scene.

HDF5 layout (N is the number of sampled frames; M is the number of source objects):

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

Frame and camera datasets share the same zero-based sampled frame index; /frames/names
maps that index to the source frame name. Each image entry stores compressed bytes.
Decoded RGB has shape (height, width, 3) in RGB order, masks have shape (height, width),
and depth has shape (192, 256) in uint16 millimetres, with zero indicating invalid depth.
Depth intrinsics are computed by scaling the RGB intrinsics to the sensor-depth grid.
Camera axes are x-right, y-down, z-forward in a mesh-aligned world with Z pointing up.
Object rows are aligned by /objects/object_id and retain the complete source population.
The objects checksum uses the sorted, compact JSON representation in object_geometry_sha256().
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import cast

import h5py
import numpy as np
from numpy.typing import NDArray

from egorecall.data.images import decode_image
from egorecall.geometry.boxes import ObjectGeometry
from egorecall.geometry.cameras import CameraSequence, scale_intrinsics

DEPTH_SIZE = (256, 192)
CACHE_VERSION = 2
IMAGE_DATASETS = {"rgb": "rgb_jpg", "depth": "depth_png", "mask": "mask_png"}
OBJECT_ARRAY_SHAPES = {"centroid": (3,), "axes": (3, 3), "lengths": (3,), "minimum": (3,), "maximum": (3,)}


@dataclass(frozen=True)
class FrameCamera:
    """
    Camera metadata for one sampled frame, with independently owned arrays.

    Args:
        frame_idx: Zero-based sampled frame index.
        frame_name: Corresponding source frame name.
        timestamp: Source timestamp in seconds.
        camera_to_world: Mesh-aligned 4x4 camera-to-world transform in metres.
        rgb_intrinsics: Pinhole intrinsics at native RGB resolution.
        depth_intrinsics: Pinhole intrinsics scaled to the sensor-depth grid.
    """

    frame_idx: int
    frame_name: str
    timestamp: float
    camera_to_world: NDArray[np.float64]
    rgb_intrinsics: NDArray[np.float64]
    depth_intrinsics: NDArray[np.float64]


@dataclass(frozen=True)
class Observation:
    """
    One RGB-D observation and its source camera, with no object annotations.

    Args:
        frame_idx: Zero-based sampled frame index.
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
    Read images, cameras, and boxes from one scene cache. Use as a context manager
    to close the HDF5 handle. Run egorecall-check before using a new or changed cache.

    Args:
        path: Prepared scene HDF5 file containing observations and source geometry.
    """

    def __init__(self, path: Path) -> None:
        """
        Open the H5 file and load frame names, camera arrays, and object geometry.
        Images stay on disk until requested. Use egorecall-check to validate the cache.

        Args:
            path: Scene-cache path.
        """
        self.path = path
        self._file = h5py.File(path, "r")
        try:
            # These small arrays are reused for every query in the scene.
            self.scene_id = self._file.attrs["scene_id"]
            self.source_fps = float(self._file.attrs["source_fps"])
            self.subsample_factor = int(self._file.attrs["subsample_factor"])
            self.image_size = tuple(int(value) for value in self._file.attrs["rgb_resolution"])
            self.frame_names = tuple(self._file["frames/names"].asstr()[:])
            self._cameras = CameraSequence(
                self.frame_names,
                self._file["camera/aligned_pose"][:],
                self._file["camera/intrinsic"][:],
                self._file["camera/timestamp"][:],
                self.image_size,
            )
            self._objects = self._load_objects()
        except BaseException:
            self._file.close()
            raise

    def _load_objects(self) -> dict[int, ObjectGeometry]:
        """
        Read the parallel object columns using their common row order.

        Returns:
            Geometry records keyed by ScanNet++ object ID.
        """
        object_group = self._file["objects"]
        object_ids = object_group["object_id"][:]
        object_labels = object_group["label"].asstr()[:]
        box_columns = {name: object_group[name][:] for name in OBJECT_ARRAY_SHAPES}
        return {
            int(object_id): ObjectGeometry(
                int(object_id),
                object_labels[position],
                *(box_columns[name][position] for name in OBJECT_ARRAY_SHAPES),
            )
            for position, object_id in enumerate(object_ids)
        }

    def objects(self) -> dict[int, ObjectGeometry]:
        """
        Read all cached source objects with independently owned geometry arrays.

        Returns:
            Object IDs, labels, and boxes, including objects outside the EgoRecall filtered population.
        """
        return {oid: object_geometry.copy() for oid, object_geometry in self._objects.items()}

    def encoded_image(self, frame_idx: int, kind: str) -> bytes:
        """
        Read JPEG or PNG bytes without decoding pixels or recomputing checksums.

        Args:
            frame_idx: Zero-based sampled frame index within the cache timeline.
            kind: One of rgb, depth, or mask.

        Returns:
            JPEG or PNG bytes suitable for independent image decoders.
        """
        dataset_name = IMAGE_DATASETS[kind]
        return self._file[f"frames/{dataset_name}"][frame_idx].tobytes()

    def camera(self, frame_idx: int) -> FrameCamera:
        """
        Return camera metadata from memory without reading or decoding image payloads.

        Args:
            frame_idx: Zero-based sampled frame index within the cache timeline.

        Returns:
            Frame identity, timestamp, pose, and RGB/depth intrinsics with fresh arrays.
        """
        intrinsic = self._cameras.intrinsics[frame_idx].copy()
        return FrameCamera(
            frame_idx=frame_idx,
            frame_name=self.frame_names[frame_idx],
            timestamp=float(self._cameras.timestamps[frame_idx]),
            camera_to_world=self._cameras.camera_to_world[frame_idx].copy(),
            rgb_intrinsics=intrinsic,
            depth_intrinsics=scale_intrinsics(intrinsic, self.image_size, DEPTH_SIZE),
        )

    def observation(self, frame_idx: int) -> Observation:
        """
        Decode one frame with independently owned image and camera arrays.

        Args:
            frame_idx: Zero-based sampled frame index within the cache timeline.

        Returns:
            Native RGB, sensor depth, anonymization mask, pose, and scaled intrinsics.
        """
        camera = self.camera(frame_idx)

        # Decode the image payloads at their native RGB and sensor-depth resolutions.
        rgb = cast(NDArray[np.uint8], decode_image(self.encoded_image(frame_idx, "rgb"), "rgb"))
        depth = cast(NDArray[np.uint16], decode_image(self.encoded_image(frame_idx, "depth"), "depth"))
        mask = cast(NDArray[np.uint8], decode_image(self.encoded_image(frame_idx, "mask"), "mask"))

        # Combine the decoded images with the independently owned camera metadata.
        return Observation(
            frame_idx=camera.frame_idx,
            frame_name=camera.frame_name,
            timestamp=camera.timestamp,
            rgb=rgb,
            depth=depth,
            mask=mask,
            camera_to_world=camera.camera_to_world,
            rgb_intrinsics=camera.rgb_intrinsics,
            depth_intrinsics=camera.depth_intrinsics,
        )

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
