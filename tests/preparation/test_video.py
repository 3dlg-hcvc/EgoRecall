"""
Extract video frames by source index with FFmpeg, and reject requests that could drop, reorder, or overwrite frames.
"""

from pathlib import Path

import numpy as np
import pytest

from egorecall.data.images import decode_image
from egorecall.preparation.video import extract_video_frames


@pytest.mark.parametrize("indices", [(7,), (1, 4, 20)])
def test_video_selection_uses_source_indices(
    raw_root: Path, tmp_path: Path, ffmpeg_path: str, indices: tuple[int, ...]
) -> None:
    """
    Select a nonzero singleton or irregular indices without shifting frame identities.

    Args:
        raw_root: Synthetic video whose green channel encodes source time.
        tmp_path: Extraction directory parent.
        ffmpeg_path: FFmpeg executable.
        indices: Exact source indices to extract.
    """
    video = raw_root / "data/scene_a/iphone/rgb.mkv"
    names = tuple(f"frame_{index:06d}" for index in indices)
    files = extract_video_frames(video, names, tmp_path / "selected", ffmpeg=ffmpeg_path)
    for index, path in zip(indices, files, strict=True):
        rgb = decode_image(path.read_bytes(), "rgb")
        np.testing.assert_allclose(rgb[2, 2], [200, 20 + index, 30], atol=5)

    with pytest.raises(ValueError, match="extracted 1 frames, expected 2"):
        extract_video_frames(video, ("frame_000000", "frame_000100"), tmp_path / "missing", ffmpeg=ffmpeg_path)


@pytest.mark.parametrize("indices", [(), (10, 0), (0, 0)])
def test_video_selection_rejected_before_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, indices: tuple[int, ...]
) -> None:
    """
    Reject selections that cannot produce one output per requested frame in the requested order.

    Args:
        tmp_path: Directory for the proposed extraction destination.
        monkeypatch: Fixture that makes any FFmpeg invocation fail the test.
        indices: Empty, descending, or repeated source indices.
    """
    from egorecall.preparation import video

    def unexpected_ffmpeg(*args: object, **kwargs: object) -> None:
        """
        Fail if an invalid extraction request reaches FFmpeg.

        Args:
            args: Subprocess positional arguments.
            kwargs: Subprocess keyword arguments.
        """
        pytest.fail("Invalid selection reached FFmpeg.")

    monkeypatch.setattr(video.subprocess, "run", unexpected_ffmpeg)
    frame_names = tuple(f"frame_{index:06d}" for index in indices)
    destination = tmp_path / "extracted"
    with pytest.raises(ValueError, match="nonempty, increasing"):
        extract_video_frames(tmp_path / "video.mkv", frame_names, destination)
    assert not destination.exists()


@pytest.mark.parametrize("filename", ["seq_000000.jpg", "notes.txt"])
def test_video_extraction_preserves_nonempty_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filename: str
) -> None:
    """
    Reject an occupied destination before FFmpeg can overwrite or mix in existing files.

    Args:
        tmp_path: Existing extraction directory.
        monkeypatch: Fixture that makes any FFmpeg invocation fail the test.
        filename: Existing output image or unrelated file to preserve.
    """
    from egorecall.preparation import video

    def unexpected_ffmpeg(*args: object, **kwargs: object) -> None:
        """
        Fail if an occupied destination reaches FFmpeg.

        Args:
            args: Subprocess positional arguments.
            kwargs: Subprocess keyword arguments.
        """
        pytest.fail("Nonempty destination reached FFmpeg.")

    monkeypatch.setattr(video.subprocess, "run", unexpected_ffmpeg)
    existing_file = tmp_path / filename
    existing_file.write_bytes(b"existing contents")
    with pytest.raises(ValueError, match="must be empty"):
        extract_video_frames(tmp_path / "video.mkv", ("frame_000000",), tmp_path)
    assert existing_file.read_bytes() == b"existing contents"
    assert list(tmp_path.iterdir()) == [existing_file]
