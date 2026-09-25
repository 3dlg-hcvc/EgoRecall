import json
import sys
from pathlib import Path

import pytest

from egorecall import DatasetPaths
from egorecall.validation.check import check_dataset
from egorecall.validation.package import verify_package
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
    Compare a split's declared supervision counts with its actual annotations and query values.

    Args:
        package_root: Synthetic annotation package.
        name: Test-split count to corrupt.
    """
    _add_manifest_hashes(package_root)
    path = package_root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["splits"]["test"]["counts"][name] += 1
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=f"manifest/splits/test/counts/{name}"):
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


def test_checker_selects_scenes_by_split_and_stage(package_root: Path, raw_root: Path, prepared_cache: Path) -> None:
    """
    Select source/cache scenes by split and stages as preparation does, while still checking the whole package.

    Args:
        package_root: Test split whose stage 1 contains scene_a only and whose stage 2 also contains scene_b.
        raw_root: Source download containing scene_a only.
        prepared_cache: Observation cache for scene_a only.
    """
    _add_manifest_hashes(package_root)
    paths = DatasetPaths(package_root, raw_root, prepared_cache)

    # Stage 1 selects only the prepared scene; every query and scene annotation is still checked.
    report = check_dataset(paths, split="test", stages=1, check_source=True, check_cache=True)
    assert report.queries_checked == 4 and report.annotation_scenes == 2
    assert report.source_scenes == report.cache_scenes == 1

    # Stage 2 also selects scene_b, which has no prepared cache.
    with pytest.raises(FileNotFoundError):
        check_dataset(paths, split="test", stages=2, check_cache=True)

    # Requested scenes must belong to the split and stage selection.
    with pytest.raises(KeyError, match="scene_b"):
        check_dataset(paths, split="test", stages=1, scene_ids=["scene_b"], check_cache=True)


@pytest.mark.parametrize(
    ("selection", "message"),
    [
        (dict(stages=1, check_cache=True), "requires a split"),
        (dict(split="test"), "only to source or cache checks"),
        (dict(scene_ids=["scene_a"]), "only to source or cache checks"),
        (dict(split="train", stages=1, check_cache=True), "unstaged"),
        (dict(split="val", check_cache=True), "not in this dataset directory"),
    ],
)
def test_invalid_checker_selection_fails(
    package_root: Path, prepared_cache: Path, selection: dict[str, object], message: str
) -> None:
    """
    Reject stages without a split, selections without source or cache checks, and splits absent from the package.

    Args:
        package_root: Test-only package with three stages.
        prepared_cache: Observation cache for scene_a.
        selection: Invalid scene-selection arguments for check_dataset().
        message: Expected error text.
    """
    _add_manifest_hashes(package_root)
    with pytest.raises(ValueError, match=message):
        check_dataset(DatasetPaths(package_root, cache_root=prepared_cache), **selection)


def test_checker_cli_selects_split_and_stages(
    package_root: Path,
    prepared_cache: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    Check the prepared scenes of a stage selection through the CLI without listing their IDs.

    Args:
        package_root: Test split whose stage 1 contains scene_a only.
        prepared_cache: Observation cache for scene_a.
        tmp_path: Directory for the configuration file.
        monkeypatch: Fixture supplying command-line arguments.
        capsys: Fixture capturing the JSON report and argument errors.
    """
    from egorecall.cli.check_dataset import main

    _add_manifest_hashes(package_root)
    config = tmp_path / "paths.toml"
    config.write_text(
        f"[paths]\ndataset_root = {json.dumps(str(package_root))}\ncache_root = {json.dumps(str(prepared_cache))}\n"
    )

    monkeypatch.setattr(
        sys, "argv", ["egorecall-check", "--config", str(config), "--cache", "--split", "test", "--stages", "1"]
    )
    main()
    report = json.loads(capsys.readouterr().out)
    assert report["queries_checked"] == 4 and report["cache_scenes"] == 1

    # Checking without annotations has no split or stage table to select scenes from.
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
            "--split",
            "test",
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "without --split or --stages" in capsys.readouterr().err
