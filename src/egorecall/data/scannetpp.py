"""
Read ScanNet++ camera records and object geometry from a user-supplied download.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from egorecall.arguments import require_integer
from egorecall.geometry.boxes import ObjectGeometry
from egorecall.geometry.cameras import CameraSequence
from egorecall.integrity import FileFingerprint, fingerprint_file
from scannetpp_common.annotations import load_annotation
from scannetpp_common.scene_release import ScannetppSceneRelease

# ScanNet++ iPhone videos have a nominal rate of 60 frames per second.
SOURCE_FPS = 60.0


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
