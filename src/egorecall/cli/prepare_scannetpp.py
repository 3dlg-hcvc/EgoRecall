"""
Prepare scene observations and object geometry from a local ScanNet++ download.
"""

import argparse
from pathlib import Path

from egorecall.config import DatasetPaths
from egorecall.data.annotations import EgoRecallAnnotations
from egorecall.data.check import select_scenes
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
    annotations = None
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

        annotations = EgoRecallAnnotations(paths.dataset_root, split=args.split, stages=args.stages)
        scene_ids = select_scenes(annotations, args.scenes)

    # Prepare each complete scene timeline, reusing compatible caches.
    for scene_id in scene_ids:
        source = ScanNetPPScene(paths.scannetpp_root, scene_id)
        print(f"Preparing or validating {scene_id}...", flush=True)

        if annotations is None:
            output = prepare_scene(source, paths.cache_root, subsample_factor=args.subsample_factor, ffmpeg=args.ffmpeg)
        else:
            metadata = annotations.get_scene(scene_id)
            output = prepare_scene(
                source,
                paths.cache_root,
                subsample_factor=metadata["subsample_factor"],
                source_fps=metadata["source_fps"],
                expected_frame_names=annotations.frame_names(scene_id),
                ffmpeg=args.ffmpeg,
            )

        print(output, flush=True)


if __name__ == "__main__":
    main()
