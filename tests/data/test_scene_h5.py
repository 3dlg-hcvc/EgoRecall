from pathlib import Path

import h5py
import numpy as np
import pytest

from egorecall import DatasetPaths
from egorecall.data import EgoRecallDataset
from egorecall.data.scene_h5 import SceneH5
from egorecall.validation.cache import validate_scene_cache


def test_camera_access_is_independent_of_images(
    package_root: Path, prepared_cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Read matching camera metadata without image access and keep returned arrays independent.

    Args:
        package_root: Query selection for the cached scene.
        prepared_cache: Scene cache containing known camera values.
        monkeypatch: Fixture used to reject any image read or decode attempt.
    """
    from egorecall.data import scene_h5

    def unexpected_image_access(*args: object, **kwargs: object) -> None:
        """
        Fail if camera access tries to read or decode an image.

        Args:
            args: Positional image-access arguments.
            kwargs: Keyword image-access arguments.
        """
        raise AssertionError("Camera access must not read or decode images.")

    with SceneH5(prepared_cache / "scene_a.h5") as cache:
        expected = cache.observation(1)
        monkeypatch.setattr(SceneH5, "encoded_image", unexpected_image_access)
        monkeypatch.setattr(scene_h5, "decode_image", unexpected_image_access)

        camera = cache.camera(1)
        assert camera.frame_idx == expected.frame_idx
        assert camera.frame_name == expected.frame_name
        assert camera.timestamp == expected.timestamp
        for name in ("camera_to_world", "rgb_intrinsics", "depth_intrinsics"):
            np.testing.assert_array_equal(getattr(camera, name), getattr(expected, name))
            getattr(camera, name)[:] = 999
            np.testing.assert_array_equal(getattr(cache.camera(1), name), getattr(expected, name))

        for invalid in (-4, 1.0, 3):
            with pytest.raises((TypeError, IndexError)):
                cache.camera(invalid)

    with EgoRecallDataset(DatasetPaths(package_root, cache_root=prepared_cache)).open_scene("scene_a") as scene_data:
        window = scene_data.query(17).observations
        camera = window.camera(1)
        np.testing.assert_array_equal(camera.camera_to_world, expected.camera_to_world)
        assert camera.frame_name == "frame_000010"

        for invalid in (-3, 1.0, 2, 100):
            with pytest.raises((TypeError, IndexError)):
                window.camera(invalid)


def test_cached_geometry_is_independently_owned(prepared_cache: Path) -> None:
    """
    Caller edits to object records must not change later cache lookups.

    Args:
        prepared_cache: Completed scene cache.
    """
    with SceneH5(prepared_cache / "scene_a.h5") as cache:
        objects = cache.objects()
        objects[1].centroid[:] = 999
        del objects[2]
        fresh = cache.objects()
        np.testing.assert_array_equal(fresh[1].centroid, [1, 0, 0])
        assert set(fresh) == {1, 2, 3}


def test_cache_reads_skip_checksum_audits(prepared_cache: Path) -> None:
    """
    Ordinary image access leaves checksum auditing to the explicit checker.

    Args:
        prepared_cache: Cache whose image bytes will stay unchanged.
    """
    path = prepared_cache / "scene_a.h5"
    with h5py.File(path, "r+") as h5_file:
        h5_file["frames/rgb_jpg_sha256"][1] = b"0" * 64
    with SceneH5(path) as scene_h5:
        assert scene_h5.observation(1).depth[0, 0] == 1010
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_scene_cache(path)
