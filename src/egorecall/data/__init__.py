"""
Dataset and scene access, annotation readers, and typed EgoRecall records.
"""

from egorecall.data.annotations import EgoRecallAnnotations, StageRange, parse_stages
from egorecall.data.dataset import EgoRecallDataset, EgoRecallScene
from egorecall.data.records import QueryRecord, SceneAnnotations, SceneRecord, decode_program

__all__ = [
    "EgoRecallAnnotations",
    "EgoRecallDataset",
    "EgoRecallScene",
    "QueryRecord",
    "SceneAnnotations",
    "SceneRecord",
    "StageRange",
    "decode_program",
    "parse_stages",
]
