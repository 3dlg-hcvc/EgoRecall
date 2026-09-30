"""
Camera and object-box records shared by source and cache access, intrinsic scaling,
and independent copies of box geometry arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from egorecall.arguments import require_integer


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
