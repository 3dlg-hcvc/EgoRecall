"""
Compute file and object-geometry fingerprints and resolve dataset-relative paths.
"""

import hashlib
import json
from pathlib import Path
from typing import TypedDict

from egorecall.geometry.boxes import ObjectGeometry


class FileFingerprint(TypedDict):
    """
    File length and SHA-256 digest, independent of the local storage path.
    """

    bytes: int
    sha256: str


def fingerprint_file(path: Path) -> FileFingerprint:
    """
    Hash a file with bounded memory.

    Args:
        path: File to read.

    Returns:
        Byte count and hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    return {"bytes": size, "sha256": digest.hexdigest()}


def relative_file(root: Path, name: str) -> Path:
    """
    Resolve a manifest filename while allowing cached-download file symlinks.

    Args:
        root: Directory containing the manifest's files.
        name: Relative filename with no parent traversal.

    Returns:
        Lexical path under root, without resolving file symlinks.
    """
    path = Path(name)
    if not name or path.is_absolute() or ".." in path.parts or path == Path("."):
        raise ValueError(f"Expected a relative file path without parent traversal: {name!r}.")
    return root / path


def object_geometry_sha256(objects_by_id: dict[int, ObjectGeometry]) -> str:
    """
    Fingerprint object IDs, labels, and box values in a deterministic representation.

    Args:
        objects_by_id: Validated source geometry keyed by object ID.

    Returns:
        SHA-256 of UTF-8 JSON with sorted IDs/keys, compact separators, and finite numbers.
    """
    records = [
        {
            "object_id": oid,
            "label": object_geometry.label,
            "centroid": object_geometry.centroid.tolist(),
            "axes": object_geometry.axes.tolist(),
            "lengths": object_geometry.lengths.tolist(),
            "minimum": object_geometry.minimum.tolist(),
            "maximum": object_geometry.maximum.tolist(),
        }
        for oid, object_geometry in sorted(objects_by_id.items())
    ]
    encoded = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
