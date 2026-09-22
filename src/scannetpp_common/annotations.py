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
    Read object labels and boxes from segGroups, indexed by ScanNet++ objectId.

    Args:
        path: Path to segments_anno.json.

    Returns:
        All source objects, including objects absent from EgoRecall annotations.
    """
    with path.open(encoding="utf-8") as stream:
        annotation = json.load(stream)
    return {object_record["objectId"]: cast(SourceObject, object_record) for object_record in annotation["segGroups"]}
