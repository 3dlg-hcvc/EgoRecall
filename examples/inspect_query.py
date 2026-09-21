"""
Inspect one query, its legal observation history, and separately requested supervision.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from egorecall import DatasetPaths
from egorecall.data import EgoRecallDataset


def main() -> None:
    """
    Print query inputs, one observation's dimensions, and optional ground-truth targets.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--stages", help="Exact stage or inclusive LO:HI range.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--query", type=int, required=True, help="Stable per-scene query_idx.")
    parser.add_argument("--supervision", action="store_true", help="Also inspect ground truth and source geometry.")
    args = parser.parse_args()

    dataset = EgoRecallDataset(DatasetPaths.from_toml(args.config), split=args.split, stages=args.stages)

    # The model-facing sample contains only query text/time and the bounded history.
    with dataset.open_scene(args.scene) as scene:
        sample = scene.query(args.query)
        frame = sample.observations.frame(sample.query.frame)

        report = {
            "query": asdict(sample.query),
            "available_frames": len(sample.observations),
            "last_frame_name": frame.frame_name,
            "timestamp": frame.timestamp,
            "rgb_shape": frame.rgb.shape,
            "depth_shape": frame.depth.shape,
            "depth_units": "millimetres",
        }

        # Ground truth is requested separately for inspection or evaluation.
        if args.supervision:
            truth = scene.supervision
            report["supervision"] = {
                "target_oids": scene.answer(args.query)["target_oids"],
                "source_objects": len(truth.source_objects),
                "filtered_objects": len(truth.filtered_objects),
            }

        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
