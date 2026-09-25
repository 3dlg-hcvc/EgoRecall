"""
Extract selected RGB and mask frames from source videos with FFmpeg.
"""

import re
import subprocess
from pathlib import Path

from egorecall.data.scannetpp import source_frame_index

# Preparation requires FFmpeg 6 or newer; older releases lack options it uses, such as -fps_mode.
# FFmpeg 6 ships libavcodec 60, whose version ffmpeg -version prints in release and development builds alike.
MINIMUM_LIBAVCODEC = 60


def require_ffmpeg(ffmpeg: str = "ffmpeg") -> None:
    """
    Require FFmpeg 6 or newer, judged by the libavcodec version that ffmpeg -version reports.

    Args:
        ffmpeg: FFmpeg executable name or path.
    """
    result = subprocess.run([ffmpeg, "-version"], check=True, capture_output=True, text=True)
    match = re.search(r"^libavcodec\s+(\d+)\.", result.stdout, re.MULTILINE)
    if match is None or int(match.group(1)) < MINIMUM_LIBAVCODEC:
        version = result.stdout.splitlines()[0] if result.stdout else "no version information"
        raise RuntimeError(f"FFmpeg 6 or newer is required; {ffmpeg} reports: {version}.")


def extract_video_frames(
    path: Path, names: tuple[str, ...], destination: Path, *, masks: bool = False, ffmpeg: str = "ffmpeg"
) -> tuple[Path, ...]:
    """
    Extract exactly the requested source indices using FFmpeg in display order.
    Native pixel orientation is retained to match the source camera intrinsics.

    Args:
        path: Source RGB or anonymization-mask video.
        names: Increasing source frame names to extract.
        destination: Empty working directory for encoded frames.
        masks: Write grayscale PNG masks instead of quality-1 JPEG RGB frames.
        ffmpeg: FFmpeg executable name or path.

    Returns:
        Encoded image paths in the same order as names.
    """
    indices = [source_frame_index(name) for name in names]
    if not indices or any(a >= b for a, b in zip(indices, indices[1:])):
        raise ValueError("Video selection requires nonempty, increasing source indices.")

    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError(f"Video extraction directory must be empty: {destination}.")

    # Use a compact stride expression when possible, with explicit first/last bounds.
    stride = indices[1] - indices[0] if len(indices) > 1 else 1
    if indices == list(range(indices[0], indices[-1] + 1, stride)):
        expression = f"between(n,{indices[0]},{indices[-1]})*not(mod(n-{indices[0]},{stride}))"
    else:
        expression = "+".join(f"eq(n,{index})" for index in indices)

    # Build and run the extraction command with the selected output encoding.
    suffix = "png" if masks else "jpg"
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-threads",
        "1",
        "-noautorotate",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-vf",
        f"select='{expression}'",
        "-fps_mode",
        "vfr",
        "-threads",
        "1",
    ]
    command += ["-pix_fmt", "gray"] if masks else ["-q:v", "1"]
    command += ["-start_number", "0", str(destination / f"seq_%06d.{suffix}")]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"FFmpeg failed for {path}:\n{error.stderr.strip()}") from error

    files = tuple(sorted(destination.glob(f"seq_*.{suffix}")))
    if len(files) != len(names):
        raise ValueError(f"{path}: extracted {len(files)} frames, expected {len(names)}.")
    return files
