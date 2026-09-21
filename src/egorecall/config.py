"""
Configure EgoRecall, ScanNet++, and cache locations and resolve relative root paths.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetPaths:
    """
    Locations of EgoRecall annotations, ScanNet++ source data, and prepared assets.
    Directly supplied relative paths resolve from the current directory.
    Configuration-file paths instead resolve from that file's directory.

    Args:
        dataset_root: EgoRecall data directory containing manifest.json.
        scannetpp_root: Optional ScanNet++ directory containing data and metadata.
        cache_root: Optional destination directory for prepared assets.
    """

    dataset_root: Path
    scannetpp_root: Path | None = None
    cache_root: Path | None = None

    def __post_init__(self) -> None:
        """
        Expand home-directory references and normalize roots to absolute paths.
        Directory existence and access permissions are not validated.
        """
        paths = (
            ("dataset_root", self.dataset_root),
            ("scannetpp_root", self.scannetpp_root),
            ("cache_root", self.cache_root),
        )
        for name, value in paths:
            if value is not None:
                if not isinstance(value, Path):
                    raise TypeError(f"{name} must be a pathlib.Path, got {type(value).__name__}.")
                object.__setattr__(self, name, Path(os.path.abspath(value.expanduser())))

    @classmethod
    def from_toml(cls, config_path: Path) -> DatasetPaths:
        """
        Read and validate a TOML paths table. Reject unknown settings and empty paths.

        Args:
            config_path: TOML file with a paths table and required dataset_root.

        Returns:
            Absolute locations, with absent optional roots represented by None.
        """
        # Read the configuration file and validate its path settings.
        config_path = config_path.expanduser().absolute()
        with config_path.open("rb") as stream:
            config = tomllib.load(stream)

        if set(config) != {"paths"} or not isinstance(config["paths"], dict):
            raise ValueError("Configuration must contain a single [paths] table.")
        paths = config["paths"]

        unknown = set(paths) - {"dataset_root", "scannetpp_root", "cache_root"}
        if unknown:
            raise ValueError(f"Unknown path settings: {sorted(unknown)}.")
        if "dataset_root" not in paths:
            raise ValueError("[paths] must specify dataset_root.")

        # Resolve each configured root relative to the configuration file's directory.
        resolved: dict[str, Path] = {}
        for name, value in paths.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty path string.")

            path = Path(value).expanduser()
            resolved[name] = path if path.is_absolute() else config_path.parent / path

        # Omitted optional roots are represented by None.
        return cls(
            dataset_root=resolved["dataset_root"],
            scannetpp_root=resolved.get("scannetpp_root"),
            cache_root=resolved.get("cache_root"),
        )
