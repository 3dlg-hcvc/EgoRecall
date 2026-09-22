"""
Check annotation files, source ScanNet++ files, and prepared caches before using them.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from egorecall.config import DatasetPaths
from egorecall.data.check import check_dataset, check_source_scenes


def main() -> None:
    """
    Run the requested checks and print a JSON report; any mismatch raises an error.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="TOML file containing [paths].")
    parser.add_argument("--scenes", nargs="+", help="Limit source/cache checks to these scenes.")
    parser.add_argument(
        "--source", action="store_true", help="Check source cameras, boxes, and their agreement with annotations."
    )
    parser.add_argument("--cache", action="store_true", help="Check prepared scene caches.")
    parser.add_argument(
        "--decode-all", action="store_true", help="Decode all cached frames instead of first/last only."
    )
    parser.add_argument(
        "--without-annotations", action="store_true", help="Check source scenes without an EgoRecall package."
    )
    parser.add_argument("--subsample-factor", type=int, help="Sampling stride with --without-annotations.")
    args = parser.parse_args()

    paths = DatasetPaths.from_toml(args.config)
    if args.without_annotations:
        if not args.scenes or args.subsample_factor is None:
            parser.error("--without-annotations requires --scenes and --subsample-factor.")
        report = check_source_scenes(
            paths, args.scenes, args.subsample_factor, check_cache=args.cache, decode_all=args.decode_all
        )
    else:
        if args.subsample_factor is not None:
            parser.error("--subsample-factor requires --without-annotations.")
        report = check_dataset(
            paths, scene_ids=args.scenes, check_source=args.source, check_cache=args.cache, decode_all=args.decode_all
        )
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
