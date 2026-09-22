"""
Check integer, text, and numeric values used by runtime operations and dataset validators.
"""

import math


def require_integer(value: object, context: str, minimum: int = 0) -> int:
    """
    Require a Python integer at or above minimum. Booleans are rejected.

    Args:
        value: Value to check.
        context: Field description used in errors.
        minimum: Smallest accepted value.

    Returns:
        The validated integer.
    """
    if type(value) is not int or value < minimum:
        raise ValueError(f"{context}: expected an integer >= {minimum}.")
    return value


def require_text(value: object, context: str) -> str:
    """
    Require a string containing at least one non-whitespace character.

    Args:
        value: Value to check.
        context: Field description used in errors.

    Returns:
        The input string, including any surrounding whitespace.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: expected a nonempty string.")
    return value


def require_number(value: object, context: str) -> None:
    """
    Require a finite, nonnegative numeric statistic. Validation leaves its value unchanged.

    Args:
        value: Numeric statistic to check.
        context: Field description used in errors.
    """
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{context}: expected a finite, nonnegative number.")
