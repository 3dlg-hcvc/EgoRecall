"""
Resolve scan and iPhone files in the original ScanNet++ scene layout.
"""

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ScannetppSceneRelease:
    """
    Locate files for one scene without opening them.

    Args:
        scene_id: Scene directory name.
        data_root: ScanNet++ data subtree containing scene directories.
    """

    scene_id: str
    data_root: Path

    def __post_init__(self) -> None:
        """
        Require a single directory name so scene paths stay under data_root.
        """
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.scene_id):
            raise ValueError(f"Invalid scene ID: {self.scene_id!r}.")

    @property
    def scene_root_dir(self) -> Path:
        """
        Locate the scene directory.

        Returns:
            Directory containing scans and iphone.
        """
        return self.data_root / self.scene_id

    @property
    def scan_mesh_path(self) -> Path:
        """
        Locate the mesh in aligned world coordinates.

        Returns:
            Path to mesh_aligned_0.05.ply.
        """
        return self.scene_root_dir / "scans/mesh_aligned_0.05.ply"

    @property
    def scan_mesh_segs_path(self) -> Path:
        """
        Locate the mesh segmentation.

        Returns:
            Path to segments.json.
        """
        return self.scene_root_dir / "scans/segments.json"

    @property
    def scan_anno_json_path(self) -> Path:
        """
        Locate object IDs, labels, segment membership, and oriented boxes.

        Returns:
            Path to segments_anno.json.
        """
        return self.scene_root_dir / "scans/segments_anno.json"

    @property
    def iphone_video_path(self) -> Path:
        """
        Locate the RGB video.

        Returns:
            Path to iphone/rgb.mkv.
        """
        return self.scene_root_dir / "iphone/rgb.mkv"

    @property
    def iphone_video_mask_path(self) -> Path:
        """
        Locate the anonymization-mask video.

        Returns:
            Path to iphone/rgb_mask.mkv.
        """
        return self.scene_root_dir / "iphone/rgb_mask.mkv"

    @property
    def iphone_depth_path(self) -> Path:
        """
        Locate the compressed sensor depth stream.

        Returns:
            Path to iphone/depth.bin.
        """
        return self.scene_root_dir / "iphone/depth.bin"

    @property
    def iphone_pose_intrinsic_imu_path(self) -> Path:
        """
        Locate camera poses, intrinsics, timestamps, and IMU records.

        Returns:
            Path to iphone/pose_intrinsic_imu.json.
        """
        return self.scene_root_dir / "iphone/pose_intrinsic_imu.json"

    @property
    def iphone_exif_path(self) -> Path:
        """
        Locate image metadata, including native pixel dimensions.

        Returns:
            Path to iphone/exif.json.
        """
        return self.scene_root_dir / "iphone/exif.json"

    @property
    def iphone_colmap_dir(self) -> Path:
        """
        Locate the iPhone COLMAP reconstruction.

        Returns:
            Path to iphone/colmap.
        """
        return self.scene_root_dir / "iphone/colmap"
