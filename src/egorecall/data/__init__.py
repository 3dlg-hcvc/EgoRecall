"""
Dataset and scene access, annotation readers, and typed EgoRecall records.
"""

from egorecall.data.annotations import EgoRecallAnnotations
from egorecall.data.dataset import EgoRecallDataset, EgoRecallScene
from egorecall.data.records import QueryRecord, SceneAnnotations, SceneRecord, decode_program
from egorecall.data.stages import StageRange, parse_stages

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
