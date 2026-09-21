"""
Verify an EgoRecall package and optional local ScanNet++ and observation-cache joins.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from egorecall.config import DatasetPaths
from egorecall.data.check import check_dataset


def main() -> None:
    """
    Run the requested checks and print a JSON report; any mismatch raises an error.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="TOML file containing [paths].")
    parser.add_argument("--scenes", nargs="+", help="Limit source/cache checks to these scenes.")
    parser.add_argument("--source", action="store_true", help="Check raw geometry, camera timelines, and object joins.")
    parser.add_argument("--cache", action="store_true", help="Check prepared scene caches.")
    parser.add_argument(
        "--decode-all", action="store_true", help="Decode all cached frames instead of first/last only."
    )
    args = parser.parse_args()

    # Apply the requested checks and report their actual coverage.
    report = check_dataset(
        DatasetPaths.from_toml(args.config),
        scene_ids=args.scenes,
        check_source=args.source,
        check_cache=args.cache,
        decode_all=args.decode_all,
    )
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
