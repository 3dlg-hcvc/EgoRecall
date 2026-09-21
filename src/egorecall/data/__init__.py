"""
Readers and record types for EgoRecall queries, frame mappings, and scene annotations.
"""

from egorecall.data.dataset import EgoRecallDataset
from egorecall.data.records import QueryRecord, SceneAnnotations, SceneRecord, decode_program
from egorecall.data.stages import StageRange, parse_stages

__all__ = [
    "EgoRecallDataset",
    "QueryRecord",
    "SceneAnnotations",
    "SceneRecord",
    "StageRange",
    "decode_program",
    "parse_stages",
]
