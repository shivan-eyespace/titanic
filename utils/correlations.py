"""Module to determine correlations."""
import math

import numpy as np


def tetrachoric(arr: np.ndarray) -> float:
    """Determine tetrachoric correlation.

    Args:
        a | b
       ---+---
        c | d

    Returns:
       (Decimal): Tetrachoric correlation
    """
    a = arr[0][0]
    b = arr[0][1]
    c = arr[1][0]
    d = arr[1][1]

    return math.cos((math.pi) / (1 + math.sqrt((b * c) / (a * d))))


def polychoric(arr: np.ndarray) -> float:
    """Determine polychoric correlation.

    Args:
        a | b
       ---+---
        c | d

    Returns:
       (Decimal): Tetrachoric correlation
    """
    a = arr[0][0]
    b = arr[0][1]
    c = arr[1][0]
    d = arr[1][1]

    return math.cos((math.pi) / (1 + math.sqrt((a * d) / (b * c))))
