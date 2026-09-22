"""
Object box records and independent copies of their geometry arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray


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
