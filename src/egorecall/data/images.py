"""
Encode and decode scene images, and check image formats during preparation and validation.
"""

from io import BytesIO

import numpy as np
from numpy.typing import NDArray
from PIL import Image


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
