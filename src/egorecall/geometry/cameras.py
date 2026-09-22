"""
Camera records and intrinsic scaling shared by source and cache access.
"""

from dataclasses import dataclass

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
