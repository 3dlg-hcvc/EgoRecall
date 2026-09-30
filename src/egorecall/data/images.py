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
    Check an encoded image's format, dimensions, channels, and completeness without decoding its pixels.
    RGB frames are JPEG, masks are 8-bit grayscale PNG, and depth is 16-bit PNG.

    Args:
        payload: Encoded image bytes.
        kind: One of rgb, depth, or mask.
        size: Expected (width, height).
    """
    with Image.open(BytesIO(payload)) as image:
        # Require the expected image dimensions, file format, and channel representation.
        if image.size != size:
            raise ValueError(f"{kind}: image dimensions {image.size} do not match {size}.")

        if kind == "rgb":
            if image.format != "JPEG" or image.mode != "RGB":
                raise ValueError(f"RGB frame must be an RGB JPEG, got {image.format}/{image.mode}.")
        elif kind == "mask":
            if image.format != "PNG" or image.mode != "L":
                raise ValueError(f"Mask frame must be an 8-bit grayscale PNG, got {image.format}/{image.mode}.")
        elif kind == "depth":
            if image.format != "PNG" or image.mode not in ("I;16", "I;16L", "I;16B"):
                raise ValueError(f"Depth frame must be 16-bit PNG, got {image.format}/{image.mode}.")
        else:
            raise ValueError(f"Unknown image kind {kind!r}.")

        # Pillow verifies PNG chunk checksums without decoding pixels. JPEG has no checksums,
        # so require the end-of-image marker, which a truncated file lacks.
        if kind == "rgb" and not payload.endswith(b"\xff\xd9"):
            raise ValueError("RGB JPEG is truncated: its end-of-image marker is missing.")
        image.verify()
