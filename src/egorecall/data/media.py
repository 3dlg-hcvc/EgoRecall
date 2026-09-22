"""
Decode selected video frames and encode depth images for local observation caches.
"""

import subprocess
from io import BytesIO
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from egorecall.data.scannetpp import source_frame_index


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


def encode_depth(depth: NDArray[np.uint16]) -> bytes:
    """
    Encode a uint16 millimetre depth map losslessly as PNG.

    Args:
        depth: Two-dimensional uint16 depth image.

    Returns:
        PNG bytes containing the unchanged depth values.
    """
    if depth.dtype != np.uint16 or depth.ndim != 2:
        raise ValueError("Depth must be a two-dimensional uint16 array.")
    with BytesIO() as buffer:
        Image.fromarray(depth).save(buffer, format="PNG")
        return buffer.getvalue()


def decode_image(payload: bytes, kind: str) -> NDArray[np.uint8] | NDArray[np.uint16]:
    """
    Decode an image into an array, retaining uint16 depth values and uint8 RGB or masks.

    Args:
        payload: Encoded JPEG or PNG bytes.
        kind: One of rgb, depth, or mask.

    Returns:
        RGB uint8 (H, W, 3), mask uint8 (H, W), or depth uint16 (H, W).
    """
    if kind not in ("rgb", "mask", "depth"):
        raise ValueError(f"Unknown image kind {kind!r}.")
    with Image.open(BytesIO(payload)) as image:
        return np.array(image, dtype=np.uint16 if kind == "depth" else np.uint8)


def validate_image(payload: bytes, kind: str, size: tuple[int, int]) -> None:
    """
    Check an encoded image's header and container without allocating decoded pixel arrays.

    Args:
        payload: Encoded image bytes.
        kind: One of rgb, depth, or mask.
        size: Expected (width, height).
    """
    with Image.open(BytesIO(payload)) as image:
        _validate_image_header(image, kind, size)
        image.verify()


def _validate_image_header(image: Image.Image, kind: str, size: tuple[int, int]) -> None:
    """
    Require the expected image dimensions and channel representation.

    Args:
        image: Image opened by Pillow.
        kind: One of rgb, depth, or mask.
        size: Expected (width, height).
    """
    if image.size != size:
        raise ValueError(f"{kind}: image dimensions {image.size} do not match {size}.")

    if kind == "rgb":
        if image.mode != "RGB":
            raise ValueError(f"RGB frame has unexpected mode {image.mode}.")
    elif kind == "mask":
        if image.mode != "L":
            raise ValueError(f"Mask frame has unexpected mode {image.mode}.")
    elif kind == "depth":
        if image.format != "PNG" or image.mode not in ("I;16", "I;16L", "I;16B"):
            raise ValueError(f"Depth frame must be 16-bit PNG, got {image.format}/{image.mode}.")
    else:
        raise ValueError(f"Unknown image kind {kind!r}.")
