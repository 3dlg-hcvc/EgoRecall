"""
Verify configuration-relative path resolution and optional root settings.
"""

from pathlib import Path

import pytest

from egorecall import DatasetPaths


def test_paths_are_relative_to_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Resolve roots relative to the configuration file even when the caller changes
    its working directory. Configured directories need not exist.

    Args:
        tmp_path: Temporary directory containing the configuration file.
        monkeypatch: Fixture used to change the working directory.
    """
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config = config_dir / "paths.toml"
    config.write_text(
        """[paths]
dataset_root = "../annotations"
scannetpp_root = "../raw/v2"
cache_root = "../prepared"
"""
    )

    monkeypatch.chdir(tmp_path.parent)
    paths = DatasetPaths.from_toml(config)
    assert paths.dataset_root == tmp_path / "annotations"
    assert paths.scannetpp_root == tmp_path / "raw/v2"
    assert paths.cache_root == tmp_path / "prepared"
    assert not paths.scannetpp_root.exists()
    assert not paths.cache_root.exists()


def test_optional_roots_can_be_omitted(tmp_path: Path) -> None:
    """
    A configuration containing only dataset_root leaves optional roots as None.

    Args:
        tmp_path: Temporary configuration directory.
    """
    config = tmp_path / "paths.toml"
    config.write_text("""[paths]
dataset_root = "annotations"
""")

    paths = DatasetPaths.from_toml(config)
    assert paths.dataset_root == tmp_path / "annotations"
    assert paths.scannetpp_root is None and paths.cache_root is None


@pytest.mark.parametrize(
    "contents",
    [
        """[paths]
dataset_root = ""
""",
        "[paths]\ndataset_root = 42",
        """[paths]
dataset_root = "data"
scannet_root = "raw"
""",
        """[other]
dataset_root = "data"
""",
    ],
)
def test_invalid_configuration_is_rejected(tmp_path: Path, contents: str) -> None:
    """
    Empty paths, wrong types, and misspelled settings must fail explicitly.

    Args:
        tmp_path: Temporary configuration directory.
        contents: Deliberately invalid TOML configuration.
    """
    config = tmp_path / "paths.toml"
    config.write_text(contents)

    with pytest.raises(ValueError):
        DatasetPaths.from_toml(config)


def test_annotation_root_can_be_omitted(tmp_path: Path) -> None:
    """
    Allow source/cache configuration for preparation without an annotation package.

    Args:
        tmp_path: Temporary configuration directory.
    """
    config = tmp_path / "paths.toml"
    config.write_text("""[paths]
scannetpp_root = "raw"
cache_root = "cache"
""")

    paths = DatasetPaths.from_toml(config)
    assert paths.dataset_root is None
    assert paths.scannetpp_root == tmp_path / "raw"
    assert paths.cache_root == tmp_path / "cache"
