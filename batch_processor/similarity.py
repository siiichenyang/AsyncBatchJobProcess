import math
from collections.abc import Sequence


def cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    if len(left) != len(right):
        raise ValueError("left and right have different dimensions")

    dot_product = sum(
        left_value * right_value
        for left_value, right_value in zip(left, right)
    )

    left_norm = math.sqrt(
        sum(value * value for value in left)
    )

    right_norm = math.sqrt(
        sum(value * value for value in right)
    )

    if left_norm == 0 or right_norm == 0:
        raise ValueError(
            f"left_norm={left_norm} or right_norm={right_norm} "
            "is zero"
        )

    return dot_product / (left_norm * right_norm)
