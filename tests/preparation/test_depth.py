"""
Decode each supported ScanNet++ depth format into uint16 millimetres.
"""

import zlib
from pathlib import Path

import lz4.block
import numpy as np
import pytest

from egorecall.preparation.depth import iter_depth_frames


@pytest.mark.parametrize("codec", ["global", "lz4", "deflate", "mixed"])
def test_depth_formats_and_units(tmp_path: Path, codec: str) -> None:
    """
    Decode supported depth formats and retain exact integer values at boundaries.

    Args:
        tmp_path: Directory for encoded depth streams.
        codec: Upstream whole-stream or per-frame compression format.
    """
    frames = np.zeros((3, 192, 256), dtype=np.uint16)
    for index in range(3):
        frames[index, :, :128] = 1000 + index * 1000
        frames[index, :, 128:] = 5000 + index * 1000

    # Encode the same depth values using each supported source representation.
    floats = frames.astype("<f4") / 1000
    if codec == "global":
        encoded = zlib.compress(floats.tobytes(), wbits=-zlib.MAX_WBITS)
    else:
        blocks = []
        for index in range(3):
            if codec == "lz4" or (codec == "mixed" and index % 2 == 0):
                block = lz4.block.compress(frames[index].tobytes(), store_size=False)
            else:
                block = zlib.compress(floats[index].tobytes(), wbits=-zlib.MAX_WBITS)
            blocks.append(len(block).to_bytes(4, "little") + block)
        encoded = b"".join(blocks)

    path = tmp_path / "depth.bin"
    path.write_bytes(encoded)

    # Check the complete sequence and a selection that preserves source indices.
    decoded = list(iter_depth_frames(path))
    assert [index for index, _ in decoded] == [0, 1, 2]
    for index, depth in decoded:
        np.testing.assert_array_equal(depth, frames[index])

    selected = list(iter_depth_frames(path, selected={1, 2}))
    assert [index for index, _ in selected] == [1, 2]

    # Truncation must fail rather than returning a partial sequence as a successful decode.
    path.write_bytes(encoded[:-5])
    with pytest.raises(ValueError):
        list(iter_depth_frames(path))


def test_whole_file_deflate_that_starts_like_a_frame_length(tmp_path: Path) -> None:
    """
    Decode a whole-file DEFLATE stream whose first four bytes form a plausible frame length. The
    supposed first frame fails to decode, so the decoder must fall back to whole-file DEFLATE.

    Args:
        tmp_path: Directory for the encoded depth stream.
    """
    frames = np.stack([np.full((192, 256), 1000 * (index + 1), dtype=np.uint16) for index in range(2)])
    raw = (frames.astype("<f4") / 1000).tobytes()

    # Uncompressed DEFLATE blocks start with a header byte and a two-byte block length. A first block
    # of 1,535 bytes makes the stream begin 00 FF 05 00, which reads as a 392,960-byte frame length.
    blocks = [raw[:1535]] + [raw[start : start + 65535] for start in range(1535, len(raw), 65535)]
    encoded = b""
    for index, block in enumerate(blocks):
        is_final = index == len(blocks) - 1
        header = bytes([is_final]) + len(block).to_bytes(2, "little") + (len(block) ^ 0xFFFF).to_bytes(2, "little")
        encoded += header + block
    assert int.from_bytes(encoded[:4], "little") == 392_960

    path = tmp_path / "depth.bin"
    path.write_bytes(encoded)
    decoded = list(iter_depth_frames(path))
    assert [index for index, _ in decoded] == [0, 1]
    for index, depth in decoded:
        np.testing.assert_array_equal(depth, frames[index])


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("too_short", "incomplete depth stream"),
        ("corrupt_frame", "invalid depth at frame 1"),
        ("truncated_prefix", "truncated length prefix at frame 2"),
    ],
)
def test_damaged_length_prefixed_depth_fails(tmp_path: Path, change: str, message: str) -> None:
    """
    Name the frame at which a length-prefixed depth stream is damaged, and reject a file too short
    to hold one length prefix.

    Args:
        tmp_path: Directory for the encoded depth stream.
        change: Damage to introduce after a valid first frame.
        message: Expected error text.
    """
    depth = np.full((192, 256), 1500, dtype=np.uint16)
    block = lz4.block.compress(depth.tobytes(), store_size=False)
    valid_frame = len(block).to_bytes(4, "little") + block
    if change == "too_short":
        encoded = b"\x00\x00"
    elif change == "corrupt_frame":
        garbage = bytes(range(256)) * 4
        encoded = valid_frame + len(garbage).to_bytes(4, "little") + garbage
    else:
        encoded = valid_frame + valid_frame + b"\x00\x00"

    path = tmp_path / "depth.bin"
    path.write_bytes(encoded)
    with pytest.raises(ValueError, match=message):
        list(iter_depth_frames(path))
