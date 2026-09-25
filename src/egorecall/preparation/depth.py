"""
Decode ScanNet++ sensor depth from whole-stream DEFLATE or per-frame LZ4/DEFLATE.
Adapted from the ScanNet++ toolkit's depth extraction; see ATTRIBUTION.md.
"""

import zlib
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO

import lz4.block
import numpy as np
from numpy.typing import NDArray

from egorecall.data.scene_h5 import DEPTH_SIZE

# The scene cache stores sensor depth without resizing, so decoded frames have its depth size.
DEPTH_WIDTH, DEPTH_HEIGHT = DEPTH_SIZE
_PIXELS = DEPTH_HEIGHT * DEPTH_WIDTH


def _millimetres(data: bytes) -> NDArray[np.uint16]:
    """
    Convert one float32 metre-depth frame to uint16 millimetres by truncation.

    Args:
        data: Exactly one little-endian float32 depth frame.

    Returns:
        Depth with shape (192, 256); zero remains the invalid-depth value.
    """
    depth = np.frombuffer(data, dtype="<f4").reshape(DEPTH_HEIGHT, DEPTH_WIDTH)
    scaled = depth * 1000
    if not np.isfinite(scaled).all() or np.any(scaled < 0) or np.any(scaled >= 65536):
        raise ValueError("Depth values cannot be represented as uint16 millimetres.")
    return scaled.astype(np.uint16)


def _decode_block(data: bytes) -> NDArray[np.uint16]:
    """
    Decode one size-prefixed payload using the two supported per-frame codecs.

    Args:
        data: Compressed payload without the four-byte length prefix.

    Returns:
        One uint16 depth image in millimetres.
    """
    try:
        decoded = lz4.block.decompress(data, uncompressed_size=_PIXELS * 2)
    except lz4.block.LZ4BlockError:
        # Older recordings store raw DEFLATE float32 metres instead of LZ4 uint16.
        decoder = zlib.decompressobj(wbits=-zlib.MAX_WBITS)
        decoded = decoder.decompress(data, _PIXELS * 4 + 1)
        if not decoder.eof or decoder.unused_data or len(decoded) != _PIXELS * 4:
            raise ValueError("Invalid per-frame DEFLATE depth payload.") from None
        return _millimetres(decoded)

    if len(decoded) != _PIXELS * 2:
        raise ValueError("Invalid per-frame LZ4 depth payload length.")
    return np.frombuffer(decoded, dtype="<u2").reshape(DEPTH_HEIGHT, DEPTH_WIDTH)


def _global_frames(stream: BinaryIO, selected: set[int] | None) -> Iterator[tuple[int, NDArray[np.uint16]]]:
    """
    Stream whole-file DEFLATE while buffering at most one decoded depth frame.

    Args:
        stream: File positioned at the start of a raw DEFLATE stream.
        selected: Source indices to return, or None for every frame.

    Returns:
        Source indices and uint16 millimetre images.
    """
    decoder = zlib.decompressobj(wbits=-zlib.MAX_WBITS)
    frame_bytes = _PIXELS * 4
    pending = b""
    frame_id = 0

    while compressed := stream.read(65536):
        while compressed:
            pending += decoder.decompress(compressed, frame_bytes - len(pending))
            compressed = decoder.unconsumed_tail
            if decoder.unused_data:
                raise ValueError("Unexpected trailing bytes after the global depth stream.")

            if len(pending) == frame_bytes:
                if selected is None or frame_id in selected:
                    yield frame_id, _millimetres(pending)
                frame_id += 1
                pending = b""

    if not decoder.eof or pending or frame_id == 0:
        raise ValueError("Truncated or incomplete global depth stream.")


def iter_depth_frames(path: Path, selected: set[int] | None = None) -> Iterator[tuple[int, NDArray[np.uint16]]]:
    """
    Decode depth frames in source order. ScanNet++ uses either whole-file raw
    DEFLATE float32 metres or length-prefixed LZ4 uint16 / DEFLATE float32 frames.
    Codec detection happens before yielding; corrupt streams raise errors.

    Args:
        path: Source iphone/depth.bin.
        selected: Source frame indices to return, or None for every frame.

    Returns:
        Iterator of source indices and (192, 256) uint16 millimetre images.
    """
    with path.open("rb") as stream:
        # A valid first block identifies the length-prefixed format. Global raw
        # DEFLATE has no magic header, so its leading bytes can resemble a length.
        header = stream.read(4)
        if len(header) != 4:
            raise ValueError(f"{path}: incomplete depth stream.")
        size = int.from_bytes(header, "little")
        first = None
        if 0 < size <= min(path.stat().st_size - 4, _PIXELS * 8):
            try:
                first = _decode_block(stream.read(size))
            except (zlib.error, ValueError):
                first = None

        # Rewind and decode the whole stream if the first block was invalid
        if first is None:
            stream.seek(0)
            try:
                yield from _global_frames(stream, selected)
            except zlib.error as error:
                raise ValueError(f"{path}: invalid DEFLATE or length-prefixed depth data.") from error
            return

        # Otherwise, the first block was valid, so decode the rest of the stream as length-prefixed frames.
        # Decode requested frames and skip other payloads while validating lengths.
        if selected is None or 0 in selected:
            yield 0, first

        file_size = path.stat().st_size
        frame_id = 1
        while header := stream.read(4):
            if len(header) != 4:
                raise ValueError(f"{path}: truncated length prefix at frame {frame_id}.")
            size = int.from_bytes(header, "little")
            if size <= 0 or size > _PIXELS * 8 or stream.tell() + size > file_size:
                raise ValueError(f"{path}: invalid payload length at frame {frame_id}.")

            if selected is None or frame_id in selected:
                try:
                    yield frame_id, _decode_block(stream.read(size))
                except (zlib.error, ValueError) as error:
                    raise ValueError(f"{path}: invalid depth at frame {frame_id}.") from error
            else:
                stream.seek(size, 1)
            frame_id += 1
