"""
Prepare canonical observations from original ScanNet++ files into a local HDF5 cache.
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np

from egorecall.data.media import encode_depth, extract_video_frames, validate_image
from egorecall.data.scannetpp import DEPTH_SIZE, ScanNetPPScene, source_frame_index
from egorecall.data.scene_h5 import CACHE_VERSION, IMAGE_DATASETS, SceneH5
from scannetpp_common.iphone import iter_depth_frames


def prepare_scene(
    source: ScanNetPPScene,
    cache_root: Path,
    *,
    subsample_factor: int = 10,
    source_fps: float = 60.0,
    expected_frame_names: tuple[str, ...] | None = None,
    ffmpeg: str = "ffmpeg",
) -> Path:
    """
    Prepare the entire canonical scene timeline and reuse only compatible caches.
    A temporary file is published atomically after validation; interrupted work
    never replaces a completed cache. Source files are read from their original paths.

    Args:
        source: Scene in the original ScanNet++ download.
        cache_root: Separate writable directory for one H5 file per scene.
        subsample_factor: Sorted pose-record stride; the benchmark uses 10.
        source_fps: Nominal source frame rate, stored as metadata only.
        expected_frame_names: Frame names from the annotation package to require,
            or None when preparing without an annotation package.
        ffmpeg: FFmpeg executable name or path.

    Returns:
        Path to the completed or validated existing observation cache.
    """
    cache_root = cache_root.expanduser().resolve()
    if cache_root.is_relative_to(source.root):
        raise ValueError("cache_root must be outside the ScanNet++ source directory.")

    if not np.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("source_fps must be finite and positive.")

    # Establish the full source timeline and check it against the benchmark frame mapping.
    cameras = source.cameras(subsample_factor)
    names = cameras.frame_names
    if expected_frame_names is not None and names != expected_frame_names:
        raise ValueError(f"{source.scene_id}: source timeline does not match the benchmark frame mapping.")

    # Reuse an existing cache only when its source files, timeline, and cameras match.
    fingerprints = source.observation_fingerprints()
    output = cache_root / f"{source.scene_id}.h5"
    if output.exists():
        with SceneH5(output) as cached:
            cached.validate_compatibility(
                source.scene_id, names, subsample_factor, source_fps, cameras=cameras, source_files=fingerprints
            )
        return output

    # Extract loose images in system temporary storage; build the H5 beside its
    # destination so publication remains atomic even when the cache is on another filesystem.
    cache_root.mkdir(parents=True, exist_ok=True)
    with (
        tempfile.TemporaryDirectory(prefix=f"egorecall-{source.scene_id}-") as temporary,
        tempfile.TemporaryDirectory(prefix=f".{source.scene_id}-", dir=cache_root) as staging,
    ):
        work = Path(temporary)
        rgb_files = extract_video_frames(source.paths.iphone_video_path, names, work / "rgb", ffmpeg=ffmpeg)
        mask_files = extract_video_frames(
            source.paths.iphone_video_mask_path, names, work / "mask", masks=True, ffmpeg=ffmpeg
        )

        # Write encoded frames one at a time, retaining native depth values and RGB orientation.
        temporary_h5 = Path(staging) / "observations.h5"
        with h5py.File(temporary_h5, "w") as cache:
            cache.attrs.update(
                format="egorecall-observations",
                schema_version=CACHE_VERSION,
                scene_id=source.scene_id,
                source_fps=source_fps,
                subsample_factor=subsample_factor,
                rgb_resolution=cameras.image_size,
                depth_resolution=DEPTH_SIZE,
                source_files=json.dumps(fingerprints, sort_keys=True),
            )

            # Allocate the timeline and encoded-image datasets before writing payloads.
            cache.create_dataset("frames/names", data=names, dtype=h5py.string_dtype("utf-8"))
            for name in IMAGE_DATASETS.values():
                cache.create_dataset(f"frames/{name}", (len(names),), dtype=h5py.vlen_dtype(np.uint8))
                cache.create_dataset(f"frames/{name}_sha256", (len(names),), dtype="S64")

            cache.create_dataset("camera/aligned_pose", data=cameras.camera_to_world)
            cache.create_dataset("camera/intrinsic", data=cameras.intrinsics)
            cache.create_dataset("camera/timestamp", data=cameras.timestamps)

            # Pack RGB and mask images in canonical frame order.
            for frame_idx, (rgb, mask) in enumerate(zip(rgb_files, mask_files, strict=True)):
                for kind, image_path in (("rgb", rgb), ("mask", mask)):
                    payload = image_path.read_bytes()
                    validate_image(payload, kind, cameras.image_size)
                    _write_image(cache, frame_idx, kind, payload)

            # Depth source indices are independent of canonical indices and can have gaps.
            position_by_source = {source_frame_index(name): position for position, name in enumerate(names)}
            remaining = set(position_by_source)
            for source_idx, depth in iter_depth_frames(
                source.paths.iphone_depth_path, selected=set(position_by_source)
            ):
                _write_image(cache, position_by_source[source_idx], "depth", encode_depth(depth))
                remaining.remove(source_idx)

            if remaining:
                raise ValueError(f"{source.scene_id}: depth is missing source frames {sorted(remaining)[:5]}.")

        # Check the finished structure, then publish without overwriting a concurrent result.
        with SceneH5(temporary_h5) as cached:
            cached.validate_compatibility(
                source.scene_id, names, subsample_factor, source_fps, cameras=cameras, source_files=fingerprints
            )
        os.link(temporary_h5, output)
    return output


def _write_image(cache: h5py.File, frame_idx: int, kind: str, payload: bytes) -> None:
    """
    Store encoded pixels with a checksum for corruption detection on later reads.

    Args:
        cache: Writable observation cache.
        frame_idx: Canonical position.
        kind: One of rgb, depth, or mask.
        payload: Complete encoded image.
    """
    name = IMAGE_DATASETS[kind]
    cache[f"frames/{name}"][frame_idx] = np.frombuffer(payload, dtype=np.uint8)
    cache[f"frames/{name}_sha256"][frame_idx] = hashlib.sha256(payload).hexdigest().encode("ascii")
