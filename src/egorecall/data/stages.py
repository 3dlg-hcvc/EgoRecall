"""
Parse evaluation stage selections. A stage is a predefined group of evaluation
queries. A number selects one stage, and LO:HI selects an inclusive range.
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class StageRange:
    """
    An inclusive range of one-based evaluation stage numbers.

    Args:
        first: First included stage.
        last: Last included stage, at least as large as first.
    """

    first: int
    last: int

    def __post_init__(self) -> None:
        """
        Reject noninteger, nonpositive, or reversed stage bounds.
        """
        if type(self.first) is not int or type(self.last) is not int or not 1 <= self.first <= self.last:
            raise ValueError("Stage bounds must be positive integers with first <= last.")


def parse_stages(value: int | str) -> StageRange:
    """
    Parse an exact stage number or an inclusive LO:HI range. For example,
    3 selects stage 3, and "1:3" selects stages 1 through 3.

    Args:
        value: A positive stage number or an inclusive range such as "1:5".

    Returns:
        The validated first and last stage numbers.
    """
    if type(value) is int:
        return StageRange(value, value)

    if not isinstance(value, str) or not re.fullmatch(r"[0-9]+(?::[0-9]+)?", value.strip()):
        raise ValueError("Use a positive stage number or an inclusive LO:HI range, such as 3 or 1:5.")

    # Reusing the first number for a singleton preserves exact-stage selection.
    parts = value.strip().split(":")
    return StageRange(int(parts[0]), int(parts[-1]))


def require_stage_range(selection: StageRange, available: tuple[int, ...]) -> None:
    """
    Require every stage in an inclusive selection to be available.

    Args:
        selection: Requested stage bounds.
        available: Stage numbers present in the assignment table.
    """
    first, last = selection.first, selection.last
    stages = set(available)
    if not stages or first < min(stages) or last > max(stages) or any(i not in stages for i in range(first, last + 1)):
        raise ValueError(f"Requested stages {first}:{last} are not all packaged; available stages: {available}.")
