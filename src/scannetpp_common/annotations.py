"""
Load ScanNet++ object records without importing rendering or model libraries.
"""

import json
from pathlib import Path
from typing import TypedDict, cast


class OrientedBox(TypedDict):
    """
    Mesh-aligned box in metres, with flattened normalizedAxes stored row by row.
    """

    centroid: list[float]
    axesLengths: list[float]
    normalizedAxes: list[float]
    min: list[float]
    max: list[float]


class SourceObject(TypedDict):
    """
    Object identity, label, mesh segments, and geometry from segments_anno.json.
    Additional upstream fields may also be present.
    """

    objectId: int
    label: str
    segments: list[int]
    obb: OrientedBox


def load_annotation(path: Path) -> dict[int, SourceObject]:
    """
    Index source object records by objectId and reject ambiguous identities.
    Geometry and segmentation values are validated by their consumers.

    Args:
        path: Path to segments_anno.json.

    Returns:
        All source objects, including objects absent from EgoRecall annotations.
    """
    with path.open(encoding="utf-8") as stream:
        annotation = json.load(stream)
    groups = annotation["segGroups"]
    if not isinstance(groups, list):
        raise ValueError(f"{path}: segGroups must be a list.")

    # Object IDs are scene-local identities; duplicate IDs must not overwrite records.
    objects: dict[int, SourceObject] = {}
    for group in groups:
        oid = group["objectId"]
        if type(oid) is not int or oid <= 0 or oid in objects:
            raise ValueError(f"{path}: invalid or duplicate objectId {oid!r}.")
        objects[oid] = cast(SourceObject, group)
    return objects
