import json
import sys
from pathlib import Path

import pytest

from egorecall import DatasetPaths
from egorecall.validation.check import check_dataset, verify_package
from tests.helpers import _add_manifest_hashes


def test_checker_source_cache_and_package(package_root: Path, raw_root: Path, prepared_cache: Path) -> None:
    """
    Validate a complete package while limiting source/cache work to one available scene.

    Args:
        package_root: Two-scene annotation package.
        raw_root: Source download containing scene_a only.
        prepared_cache: Observation cache for scene_a.
    """
    _add_manifest_hashes(package_root)
    report = check_dataset(
        DatasetPaths(package_root, raw_root, prepared_cache),
        scene_ids=["scene_a"],
        check_source=True,
        check_cache=True,
        decode_all=True,
    )
    assert report.queries_checked == 4
    assert report.annotation_scenes == 2
    assert report.source_scenes == report.cache_scenes == 1
    assert report.frames_decoded == 3

    # Unlisted required payloads cannot bypass integrity verification.
    path = package_root / "manifest.json"
    manifest = json.loads(path.read_text())
    del manifest["files"]["queries/test.parquet"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="omits required files"):
        verify_package(package_root)

    # A listed payload must still match its recorded checksum after all required files are restored.
    _add_manifest_hashes(package_root)
    with (package_root / "scenes.json").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_package(package_root)


@pytest.mark.parametrize("name", ["objects", "any_target_queries"])
def test_checker_annotation_counts_fail(package_root: Path, name: str) -> None:
    """
    Compare manifest supervision totals with actual annotations and query values.

    Args:
        package_root: Synthetic annotation package.
        name: Manifest total to corrupt.
    """
    _add_manifest_hashes(package_root)
    path = package_root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["counts"][name] += 1
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=f"manifest/counts/{name}"):
        check_dataset(DatasetPaths(package_root))


def test_source_checker_without_annotation_package(
    raw_root: Path,
    prepared_cache: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    Validate source inputs and a prepared cache through the CLI without any dataset_root.

    Args:
        raw_root: Synthetic ScanNet++ source.
        prepared_cache: Matching scene cache.
        tmp_path: Directory for the configuration file.
        monkeypatch: Fixture supplying command-line arguments.
        capsys: Fixture capturing the JSON report.
    """
    from egorecall.cli.check_dataset import main

    config = tmp_path / "source_only.toml"
    config.write_text(f"""[paths]
scannetpp_root = "{raw_root}"
cache_root = "{prepared_cache}"
""")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "egorecall-check",
            "--config",
            str(config),
            "--without-annotations",
            "--scenes",
            "scene_a",
            "--subsample-factor",
            "10",
            "--cache",
            "--decode-all",
        ],
    )
    main()
    report = json.loads(capsys.readouterr().out)
    assert report == dict(
        files_verified=0, queries_checked=0, annotation_scenes=0, source_scenes=1, cache_scenes=1, frames_decoded=3
    )
