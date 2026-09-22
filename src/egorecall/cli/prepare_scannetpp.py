"""
Prepare scene observations and object geometry from a local ScanNet++ download.
"""

import argparse
from pathlib import Path

from egorecall.config import DatasetPaths
from egorecall.data.metadata import load_scene_metadata
from egorecall.data.prepare import prepare_scene
from egorecall.data.scannetpp import ScanNetPPScene


def main() -> None:
    """
    Prepare scenes with or without an annotation package and report cache paths.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="TOML file containing [paths].")
    parser.add_argument("--split", choices=("train", "val", "test"), help="Annotation-package split to prepare.")
    parser.add_argument("--stages", help="Exact stage or inclusive LO:HI range.")
    parser.add_argument("--scenes", nargs="+", help="Scene IDs to prepare.")
    parser.add_argument(
        "--without-annotations", action="store_true", help="Prepare source scenes without an annotation package."
    )
    parser.add_argument("--subsample-factor", type=int, help="Required sampling stride with --without-annotations.")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="FFmpeg executable name or path.")
    args = parser.parse_args()

    # Validate preparation roots before selecting scenes.
    paths = DatasetPaths.from_toml(args.config)
    if paths.scannetpp_root is None or paths.cache_root is None:
        parser.error("Preparation requires scannetpp_root and cache_root in the configuration.")

    # Use scene selection and sampling settings from the annotation package,
    # or take them directly from the command line when preparing without annotations.
    metadata_by_scene = None
    if args.without_annotations:
        if not args.scenes or args.subsample_factor is None or args.split is not None or args.stages is not None:
            parser.error("--without-annotations requires --scenes and --subsample-factor, without --split or --stages.")
        if len(args.scenes) != len(set(args.scenes)):
            parser.error("--scenes must not contain duplicate IDs.")
        scene_ids = tuple(args.scenes)
    else:
        if args.split is None or args.subsample_factor is not None:
            parser.error(
                "Preparation with annotations requires --split; its sampling stride comes from scene metadata."
            )
        if paths.dataset_root is None:
            parser.error("Preparation with annotations requires dataset_root in the configuration.")

        metadata_by_scene = load_scene_metadata(
            paths.dataset_root, args.split, stages=args.stages, scene_ids=args.scenes
        )
        scene_ids = tuple(metadata_by_scene)

    # Prepare all frames for each selected scene, leaving existing cache files in place.
    for scene_id in scene_ids:
        source_scene = ScanNetPPScene(paths.scannetpp_root, scene_id)
        print(f"Preparing {scene_id}...", flush=True)

        if metadata_by_scene is None:
            output = prepare_scene(
                source_scene, paths.cache_root, subsample_factor=args.subsample_factor, ffmpeg=args.ffmpeg
            )
        else:
            scene_meta = metadata_by_scene[scene_id]
            scene_record = scene_meta.scene_record
            output = prepare_scene(
                source_scene,
                paths.cache_root,
                subsample_factor=scene_record["subsample_factor"],
                source_fps=scene_record["source_fps"],
                frame_names=scene_meta.frame_names,
                ffmpeg=args.ffmpeg,
            )

        print(output, flush=True)


if __name__ == "__main__":
    main()
