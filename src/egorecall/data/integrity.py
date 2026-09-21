"""
Compute file fingerprints and validate relative paths in dataset manifests.
"""

import hashlib
from pathlib import Path
from typing import TypedDict


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
