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
    with dataset.open_scene(args.scene) as scene_data:
        sample = scene_data.query(args.query)
        observation = sample.observations.frame(sample.query.frame)

        report = {
            "query": asdict(sample.query),
            "available_frames": len(sample.observations),
            "last_frame_name": observation.frame_name,
            "timestamp": observation.timestamp,
            "rgb_shape": observation.rgb.shape,
            "depth_shape": observation.depth.shape,
            "depth_units": "millimetres",
        }

        # Ground truth is requested separately for inspection or evaluation.
        if args.supervision:
            scene_supervision = scene_data.supervision
            report["supervision"] = {
                "target_oids": scene_data.answer(args.query)["target_oids"],
                "source_objects": len(scene_supervision.source_objects),
                "filtered_objects": len(scene_supervision.filtered_objects),
            }

        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
