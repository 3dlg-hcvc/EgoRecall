"""
Helpers for constructing local test datasets.
"""

import json
from pathlib import Path

from egorecall.integrity import fingerprint_file


def add_manifest_hashes(root: Path) -> None:
    """
    Fingerprint synthetic package files for checksum-checker tests.

    Args:
        root: Synthetic annotation package.
    """
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"] = {
        str(file.relative_to(root)): fingerprint_file(file)
        for file in root.rglob("*")
        if file.is_file() and file != path
    }
    path.write_text(json.dumps(manifest))
