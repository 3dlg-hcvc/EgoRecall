"""
Check split names, counts, and stage ranges declared in manifest.json.
"""

from typing import cast

from egorecall.arguments import require_integer
from egorecall.data.records import SplitManifest


def manifest_splits(manifest: dict[str, object]) -> dict[str, SplitManifest]:
    """
    Check split names and stage ranges, then return each split's counts and range.
    Schema 1 stores one split under selection; schema 2 uses a splits mapping.

    Args:
        manifest: JSON object read from manifest.json.

    Returns:
        Records keyed by train, val, or test, each containing counts and an inclusive
        first/last stage range. Training records have stages set to None.
    """
    version = require_integer(manifest["schema_version"], "manifest/schema_version", minimum=1)
    if version == 1:
        selection = manifest["selection"]
        if not isinstance(selection, dict):
            raise ValueError("manifest/selection must contain a JSON object.")
        split = selection["split"]
        stages = None if split == "train" else {"first": selection["stage_from"], "last": selection["stage_to"]}
        splits = {split: {"counts": manifest["counts"], "stages": stages}}
    elif version == 2:
        if not isinstance(manifest["counts"], dict):
            raise ValueError("manifest/counts must contain a JSON object.")
        splits = manifest["splits"]
        if not isinstance(splits, dict) or not splits:
            raise ValueError("manifest/splits must contain a nonempty split mapping.")
    else:
        raise ValueError(f"Unsupported package schema_version: {version}.")

    # Training has no stage assignments; validation and test need an inclusive stage range.
    for split, split_manifest in splits.items():
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split {split!r}.")
        if not isinstance(split_manifest["counts"], dict):
            raise ValueError("manifest/counts must contain a JSON object.")
        stages = split_manifest["stages"]
        if split == "train":
            if stages is not None:
                raise ValueError("Training must be unstaged.")
        else:
            if not isinstance(stages, dict):
                raise ValueError(f"{split}: manifest stages must contain first/last bounds.")
            first = require_integer(stages["first"], "manifest/stage_from", minimum=1)
            require_integer(stages["last"], "manifest/stage_to", minimum=first)
    return cast(dict[str, SplitManifest], splits)
