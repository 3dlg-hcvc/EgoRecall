from pathlib import Path

import numpy as np

from egorecall.data.scannetpp import ScanNetPPScene
from egorecall.geometry.cameras import scale_intrinsics


def test_raw_geometry_and_camera_conventions(raw_root: Path) -> None:
    """
    Preserve source object membership, matrix orientation, timestamps, and intrinsic scaling.

    Args:
        raw_root: Scene with known source boxes and cameras.
    """
    source_scene = ScanNetPPScene(raw_root, "scene_a")
    cameras = source_scene.cameras()
    assert cameras.frame_names == ("frame_000000", "frame_000010", "frame_000020")
    np.testing.assert_array_equal(cameras.camera_to_world[:, 0, 3], [0, 0.1, 0.2])
    np.testing.assert_allclose(cameras.timestamps, 100 + np.array([0, 10, 20]) / 60)

    original = cameras.intrinsics.copy()
    scaled = scale_intrinsics(cameras.intrinsics, (32, 24), (256, 192))
    np.testing.assert_array_equal(scaled[0], [[160, 0, 128], [0, 160, 96], [0, 0, 1]])
    np.testing.assert_array_equal(cameras.intrinsics, original)

    objects = source_scene.objects()
    assert set(objects) == {1, 2, 3}
    np.testing.assert_array_equal(objects[2].lengths, [1, 2, 3])
    assert objects[3].label == "lamp"

    # The scene locates every source file that preparation reads in the original layout.
    assert all(path.is_file() for path in source_scene.cache_sources().values())
