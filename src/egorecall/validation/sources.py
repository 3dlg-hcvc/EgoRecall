"""
Check raw ScanNet++ file availability, camera values, and object geometry before preparation.
"""

import json

import numpy as np

from egorecall.arguments import require_integer, require_text
from egorecall.data.records import SceneAnnotations
from egorecall.data.scannetpp import ScanNetPPScene, source_frame_index
from egorecall.geometry import CameraSequence, ObjectGeometry


def validate_source_scene(
    source_scene: ScanNetPPScene, subsample_factor: int
) -> tuple[CameraSequence, dict[int, ObjectGeometry]]:
    """
    Check the source files used to build an H5 cache. Video and compressed depth
    payloads are decoded during preparation; this checks their presence plus the
    camera and object records needed to interpret them.

    Args:
        source_scene: Scene in a local ScanNet++ download.
        subsample_factor: Number of source pose records between sampled frames.

    Returns:
        Checked camera records and object geometry for source/cache comparisons.
    """
    require_integer(subsample_factor, "subsample_factor", minimum=1)
    for path in source_scene.cache_sources().values():
        if not path.is_file():
            raise FileNotFoundError(path)

    # Read IDs before creating a dictionary, where duplicate IDs would overwrite each other.
    with source_scene.scan_anno_json_path.open(encoding="utf-8") as stream:
        object_records = json.load(stream)["segGroups"]
    if not isinstance(object_records, list):
        raise ValueError("segGroups must be a list.")
    object_ids: set[int] = set()
    for object_record in object_records:
        object_id = require_integer(object_record["objectId"], "objectId", minimum=1)
        if object_id in object_ids:
            raise ValueError(f"Duplicate source objectId: {object_id}.")
        object_ids.add(object_id)
        require_text(object_record["label"], f"{object_id}/label")

    with source_scene.iphone_exif_path.open(encoding="utf-8") as stream:
        exif_records = json.load(stream)
    image_sizes = {
        (exif_record["PixelXDimension"], exif_record["PixelYDimension"]) for exif_record in exif_records.values()
    }
    if len(image_sizes) != 1:
        raise ValueError("EXIF must specify one consistent RGB resolution.")

    camera_sequence = source_scene.cameras(subsample_factor)
    validate_cameras(camera_sequence)
    source_objects = source_scene.objects()
    for object_id, object_geometry in source_objects.items():
        validate_object_geometry(object_geometry, f"{source_scene.scene_id}/{object_id}")
    return camera_sequence, source_objects


def validate_object_annotations(
    scene_id: str, source_objects: dict[int, ObjectGeometry], scene_annotations: SceneAnnotations
) -> None:
    """
    Check that every annotated object has geometry with the same ID and label.

    Args:
        scene_id: Scene being checked.
        source_objects: Full object set from the source or an H5 cache.
        scene_annotations: EgoRecall object visibility data for the same scene.
    """
    if scene_annotations["scene_id"] != scene_id:
        raise ValueError("Geometry and annotation scenes differ.")

    # List every annotated object without geometry, rather than failing on the first missing ID.
    missing = sorted({int(object_id) for object_id in scene_annotations["objects"]} - source_objects.keys())
    if missing:
        raise ValueError(f"{scene_id}: annotated objects have no ScanNet++ geometry: {missing}.")

    for object_id, object_annotation in scene_annotations["objects"].items():
        if source_objects[int(object_id)].label != object_annotation["label"]:
            raise ValueError(f"{scene_id}/{object_id}: source and annotation labels differ.")


def validate_object_geometry(object_geometry: ObjectGeometry, context: str) -> None:
    """
    Check an object's box shapes, finite values, extents, and orthonormal axes.

    Args:
        object_geometry: Object geometry from a source annotation or scene cache.
        context: Scene/object description used in errors.
    """
    for vector in (object_geometry.centroid, object_geometry.lengths, object_geometry.minimum, object_geometry.maximum):
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError(f"{context}: invalid bounding-box vectors.")

    if np.any(object_geometry.lengths < 0) or np.any(object_geometry.maximum < object_geometry.minimum):
        raise ValueError(f"{context}: invalid bounding-box extents.")

    if (
        object_geometry.axes.shape != (3, 3)
        or not np.isfinite(object_geometry.axes).all()
        or not np.allclose(object_geometry.axes @ object_geometry.axes.T, np.eye(3), atol=1e-4)
    ):
        raise ValueError(f"{context}: box axes must be orthonormal rows.")


def validate_cameras(camera_sequence: CameraSequence) -> None:
    """
    Validate camera array shapes, finite values, and source ordering.
    Preserve the recorded transforms, including small floating-point deviations.

    Args:
        camera_sequence: Poses, intrinsics, and timestamps in sampled frame order.
    """
    count = len(camera_sequence.frame_names)
    indices = [source_frame_index(name) for name in camera_sequence.frame_names]
    if not count or any(a >= b for a, b in zip(indices, indices[1:])):
        raise ValueError("Camera frame names must be nonempty, unique, and ordered by source index.")

    # All camera arrays must describe the same timeline with finite values.
    for array, shape in (
        (camera_sequence.camera_to_world, (count, 4, 4)),
        (camera_sequence.intrinsics, (count, 3, 3)),
        (camera_sequence.timestamps, (count,)),
    ):
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f"Camera data must be finite with shape {shape}.")

    # Validate homogeneous poses and right-handed, orthonormal rotations.
    if not np.allclose(camera_sequence.camera_to_world[:, 3, :], [0, 0, 0, 1], atol=1e-4):
        raise ValueError("Camera poses must be homogeneous camera-to-world transforms.")
    rotations = camera_sequence.camera_to_world[:, :3, :3]
    if not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-3):
        raise ValueError("Camera rotations must be orthonormal.")
    if not np.allclose(np.linalg.det(rotations), 1.0, atol=1e-3):
        raise ValueError("Camera rotations must preserve handedness.")

    if not np.allclose(camera_sequence.intrinsics[:, 2, :], [0, 0, 1]):
        raise ValueError("Intrinsics must be pinhole matrices.")
    if np.any(camera_sequence.intrinsics[:, (0, 1), (0, 1)] <= 0):
        raise ValueError("Focal lengths must be positive.")

    if np.any(np.diff(camera_sequence.timestamps) <= 0):
        raise ValueError("Camera timestamps must be strictly increasing.")

    for dimension in camera_sequence.image_size:
        require_integer(dimension, "RGB dimension", minimum=1)
