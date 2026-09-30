"""
Check encoded scene images by format, size, channels, and completeness without decoding their pixels.
"""

from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from egorecall.data.images import encode_depth, validate_image


def encode_image(pixels: np.ndarray, image_format: str) -> bytes:
    """
    Encode a synthetic image with Pillow.

    Args:
        pixels: uint8 RGB values with shape (H, W, 3), or grayscale values with shape (H, W).
        image_format: Pillow format name, such as JPEG or PNG.

    Returns:
        The encoded image bytes.
    """
    with BytesIO() as buffer:
        Image.fromarray(pixels).save(buffer, format=image_format)
        return buffer.getvalue()


RGB = np.random.default_rng(0).integers(0, 256, (24, 32, 3), dtype=np.uint8)
MASK = np.full((24, 32), 255, dtype=np.uint8)
RGB_JPEG = encode_image(RGB, "JPEG")


def test_expected_images_pass() -> None:
    """
    Accept an RGB JPEG, a grayscale PNG mask, and a 16-bit PNG depth map of the expected sizes.
    """
    validate_image(RGB_JPEG, "rgb", (32, 24))
    validate_image(encode_image(MASK, "PNG"), "mask", (32, 24))
    validate_image(encode_depth(np.full((192, 256), 1500, dtype=np.uint16)), "depth", (256, 192))


@pytest.mark.parametrize(
    ("payload", "kind", "message"),
    [
        (RGB_JPEG[:-10], "rgb", "truncated"),
        (encode_image(RGB, "PNG"), "rgb", "must be an RGB JPEG"),
        (encode_image(MASK, "JPEG"), "mask", "8-bit grayscale PNG"),
        (encode_image(RGB, "PNG"), "mask", "8-bit grayscale PNG"),
    ],
    ids=["truncated JPEG", "PNG as RGB", "JPEG as mask", "RGB PNG as mask"],
)
def test_wrong_or_truncated_images_fail(payload: bytes, kind: str, message: str) -> None:
    """
    Reject a truncated JPEG and images whose file format or channels do not match their kind,
    even though their headers give the expected size.

    Args:
        payload: Encoded image to check.
        kind: Image kind the payload is checked as.
        message: Expected error text.
    """
    with pytest.raises(ValueError, match=message):
        validate_image(payload, kind, (32, 24))
