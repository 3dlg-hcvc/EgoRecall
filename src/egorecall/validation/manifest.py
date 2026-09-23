"""
Check split names, counts, and stage ranges declared in manifest.json.
"""

from typing import cast

from egorecall.arguments import require_integer
from egorecall.data.records import SplitManifest


def manifest_splits(manifest: dict[str, object]) -> dict[str, SplitManifest]:
    """
    Check the splits mapping in manifest.json, then return each split's counts and stage range.
    The manifest lists each split under splits and the whole dataset's totals under counts.

    Args:
        manifest: JSON object read from manifest.json.

    Returns:
        Records keyed by train, val, or test, each containing counts and an inclusive
        first/last stage range. Training records have stages set to None.
    """
    version = require_integer(manifest["schema_version"], "manifest/schema_version", minimum=1)
    if version != 2:
        raise ValueError(f"Unsupported package schema_version: {version}.")
    if not isinstance(manifest["counts"], dict):
        raise ValueError("manifest/counts must contain a JSON object.")
    splits = manifest["splits"]
    if not isinstance(splits, dict) or not splits:
        raise ValueError("manifest/splits must contain a nonempty split mapping.")

    # Training has no stage assignments; validation and test need an inclusive stage range.
    for split, split_manifest in splits.items():
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split {split!r}.")
        if not isinstance(split_manifest["counts"], dict):
            raise ValueError(f"manifest/splits/{split}/counts must contain a JSON object.")
        stages = split_manifest["stages"]
        if split == "train":
            if stages is not None:
                raise ValueError("Training must be unstaged.")
        else:
            if not isinstance(stages, dict):
                raise ValueError(f"{split}: manifest stages must contain first/last bounds.")
            first = require_integer(stages["first"], f"manifest/splits/{split}/stages/first", minimum=1)
            require_integer(stages["last"], f"manifest/splits/{split}/stages/last", minimum=first)
    return cast(dict[str, SplitManifest], splits)
