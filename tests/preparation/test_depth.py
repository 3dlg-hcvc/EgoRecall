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
