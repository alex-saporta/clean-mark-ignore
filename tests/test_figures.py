"""Figure helpers that do not need any result."""

import numpy as np

from cmi.figures import FIGURE_FUNCTIONS, spread


def test_end_labels_keep_their_order_and_never_overlap():
    values = [-1.9, -2.0, -1.7, -1.4, -0.4]
    placed = spread(values, gap=0.3)
    assert list(np.argsort(placed)) == list(np.argsort(values))
    assert np.diff(np.sort(placed)).min() >= 0.3 - 1e-9
    assert spread([0.0, 5.0], gap=1.0) == [0.0, 5.0]


def test_each_figure_has_its_own_command():
    assert set(FIGURE_FUNCTIONS) == {"map", "threshold", "detectors", "movies"}
