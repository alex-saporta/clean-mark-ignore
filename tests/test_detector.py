import math

import numpy as np
import pytest

from cmi.detector import detection_quality, detector_noise, flag_cells
from cmi.strategies import corrupt

N_ROWS = 2000
N_COLS = 5


@pytest.fixture
def errors():
    X = np.random.default_rng(1).normal(size=(N_ROWS, N_COLS))
    return corrupt(X, X.copy(), 0.2, split=0).train_mask


@pytest.fixture
def noise():
    return detector_noise((N_ROWS, N_COLS), split=0, part="train")


@pytest.mark.parametrize("recall", [0.0, 0.25, 0.5, 0.9, 1.0])
@pytest.mark.parametrize("fpr", [0.0, 0.01, 0.05, 0.2, 1.0])
def test_each_column_reaches_the_requested_recall_and_fpr(errors, noise, recall, fpr):
    flagged = flag_cells(errors, recall, fpr, noise)
    for col in range(N_COLS):
        is_error = errors[:, col]
        assert (flagged[:, col] & is_error).sum() == round(recall * is_error.sum())
        assert (flagged[:, col] & ~is_error).sum() == round(fpr * (~is_error).sum())


def test_perfect_detector_flags_exactly_the_errors(errors, noise):
    assert np.array_equal(flag_cells(errors, 1.0, 0.0, noise), errors)


@pytest.mark.parametrize(
    ("low", "high"),
    [((0.5, 0.05), (0.9, 0.05)), ((0.5, 0.01), (0.5, 0.2))],
)
def test_detectors_are_nested(errors, noise, low, high):
    weak = flag_cells(errors, *low, noise)
    strong = flag_cells(errors, *high, noise)
    assert np.all(strong[weak])
    assert strong.sum() > weak.sum()


def test_noise_is_reproducible_and_differs_across_splits_and_parts():
    shape = (N_ROWS, N_COLS)
    first = detector_noise(shape, split=0, part="train")
    assert np.array_equal(first, detector_noise(shape, split=0, part="train"))
    assert not np.array_equal(first, detector_noise(shape, split=1, part="train"))
    assert not np.array_equal(first, detector_noise(shape, split=0, part="test"))


def test_a_part_without_errors_only_gets_false_alarms(noise):
    clean = np.zeros((N_ROWS, N_COLS), dtype=bool)
    flagged = flag_cells(clean, 0.9, 0.05, noise)
    assert (flagged.sum(axis=0) == round(0.05 * N_ROWS)).all()
    quality = detection_quality(clean, flagged)
    assert math.isnan(quality.recall)
    assert quality.precision == 0.0


def test_measured_quality_matches_the_expected_precision(errors, noise):
    quality = detection_quality(errors, flag_cells(errors, 0.9, 0.05, noise))
    assert quality.recall == pytest.approx(0.9, abs=1e-3)
    assert quality.fpr == pytest.approx(0.05, abs=1e-3)
    # With 20 % of errors: 0.9 * 0.2 / (0.9 * 0.2 + 0.05 * 0.8)
    assert quality.precision == pytest.approx(0.18 / 0.22, abs=1e-3)


def test_precision_depends_on_the_error_rate(noise):
    X = np.random.default_rng(1).normal(size=(N_ROWS, N_COLS))
    low = corrupt(X, X.copy(), 0.05, split=0).train_mask
    high = corrupt(X, X.copy(), 0.20, split=0).train_mask
    precision_low = detection_quality(low, flag_cells(low, 0.9, 0.05, noise)).precision
    precision_high = detection_quality(high, flag_cells(high, 0.9, 0.05, noise)).precision
    assert precision_low < precision_high


def test_no_flag_gives_undefined_precision(errors, noise):
    quality = detection_quality(errors, flag_cells(errors, 0.0, 0.0, noise))
    assert quality.recall == 0.0
    assert math.isnan(quality.precision)


def test_inputs_are_not_modified(errors, noise):
    errors_before, noise_before = errors.copy(), noise.copy()
    flag_cells(errors, 0.5, 0.1, noise)
    assert np.array_equal(errors, errors_before)
    assert np.array_equal(noise, noise_before)


@pytest.mark.parametrize(("recall", "fpr"), [(-0.1, 0.0), (1.1, 0.0), (0.5, -0.1), (0.5, 1.1)])
def test_out_of_range_values_are_rejected(errors, noise, recall, fpr):
    with pytest.raises(ValueError):
        flag_cells(errors, recall, fpr, noise)


def test_shape_mismatch_is_rejected(errors):
    with pytest.raises(ValueError):
        flag_cells(errors, 0.5, 0.1, np.zeros((3, 3)))
