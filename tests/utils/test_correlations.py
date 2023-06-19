"""Testing for correlations module."""

import numpy as np

from utils.correlations import tetrachoric


def test_tetrachoric():
    """Ensure tetrachoric function works."""
    arr = np.array([[13, 32], [17, 23]])
    result = tetrachoric(arr)
    assert result < 0.235 and result >= 0.23
