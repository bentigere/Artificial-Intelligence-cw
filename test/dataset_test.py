
import math

import numpy as np
import pytest

from neural_network.dataset import load_datasets
from genetic_algorithm.chromosome import LOWER_LIMITS, UPPER_LIMITS


@pytest.fixture
def datasets():
    """Load the existing MATLAB-compatible GA dataset."""
    return load_datasets()


def test_dataset_split_sizes(datasets):
    train, val, test = datasets

    assert len(train) == 210
    assert len(val) == 45
    assert len(test) == 45


def test_input_output_dimensions(datasets):
    train, val, test = datasets

    for inputs, targets in train + val + test:
        assert len(inputs) == 2
        assert len(targets) == 24


def test_input_features_on_unit_circle(datasets):
    train, val, test = datasets

    for inputs, _ in train + val + test:
        sin_phase, cos_phase = inputs

        assert math.isclose(
            sin_phase**2 + cos_phase**2,
            1.0,
            abs_tol=1e-10,
        )


def test_targets_are_finite(datasets):
    train, val, test = datasets

    targets = np.asarray(
        [angles for _, angles in train + val + test],
        dtype=float,
    )

    assert np.isfinite(targets).all()


def test_joint_angles_within_limits(datasets):
    train, val, test = datasets

    targets = np.asarray(
        [angles for _, angles in train + val + test],
        dtype=float,
    )

    assert np.all(targets >= LOWER_LIMITS)
    assert np.all(targets <= UPPER_LIMITS)


def test_split_is_reproducible():
    first = load_datasets()
    second = load_datasets()

    assert first == second


def test_no_duplicate_samples_between_splits(datasets):
    train, val, test = datasets

    # Phase inputs identify the original 300 gait frames.
    def phase_keys(samples):
        return {tuple(inputs) for inputs, _ in samples}

    train_keys = phase_keys(train)
    val_keys = phase_keys(val)
    test_keys = phase_keys(test)

    assert len(train_keys) == 210
    assert len(val_keys) == 45
    assert len(test_keys) == 45

    assert train_keys.isdisjoint(val_keys)
    assert train_keys.isdisjoint(test_keys)
    assert val_keys.isdisjoint(test_keys)
