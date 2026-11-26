"""Tests for the in-memory neural network evaluation functions."""

import numpy as np
import pytest

from genetic_algorithm.chromosome import LOWER_LIMITS, UPPER_LIMITS
from neural_network.updated_evaluate import (
    updated_evaluate,
    phase_inputs,
    validate_gait,
    cycle_metrics,
)


class StubModel:
    """Return controlled predictions to test evaluation independently."""

    def __init__(self, predictions):
        self.predictions = predictions
        self.received_inputs = None

    def predict(self, X):
        self.received_inputs = X
        return self.predictions


def test_phase_inputs_are_ordered_periodic_features():
    inputs = phase_inputs()
    phases = 2 * np.pi * np.arange(300) / 300

    assert inputs.shape == (300, 2)
    assert np.isfinite(inputs).all()

    np.testing.assert_allclose(inputs[:, 0], np.sin(phases))
    np.testing.assert_allclose(inputs[:, 1], np.cos(phases))
    np.testing.assert_allclose(inputs[0], [0.0, 1.0])

    # The endpoint is excluded, avoiding a duplicated first frame.
    assert not np.array_equal(inputs[-1], inputs[0])


def test_metrics_match_hand_calculation():
    inputs = np.zeros((2, 2))
    targets = np.zeros((2, 24))
    predictions = np.zeros_like(targets)
    predictions[:, 0] = [1.0, -3.0]

    model = StubModel(predictions)
    metrics = updated_evaluate(model, inputs, targets)

    assert model.received_inputs is inputs
    assert metrics["sample_count"] == 2
    assert metrics["mse"] == pytest.approx(5.0 / 24)
    assert metrics["mae"] == pytest.approx(2.0 / 24)

    np.testing.assert_allclose(metrics["per_joint_mse"], [5.0] + [0.0] * 23)
    np.testing.assert_allclose(metrics["per_joint_mae"], [2.0] + [0.0] * 23)


def test_identical_predictions_have_zero_error():
    targets = np.arange(72, dtype=float).reshape(3, 24) / 100
    metrics = updated_evaluate(
        StubModel(targets.copy()),
        np.zeros((3, 2)),
        targets,
    )

    assert metrics["mse"] == 0.0
    assert metrics["mae"] == 0.0
    np.testing.assert_array_equal(metrics["per_joint_mse"], np.zeros(24))
    np.testing.assert_array_equal(metrics["per_joint_mae"], np.zeros(24))


@pytest.mark.parametrize(
    "predictions, targets",
    [
        (np.zeros((2, 23)), np.zeros((2, 23))),
        (np.zeros((2, 24)), np.zeros((3, 24))),
        (np.zeros((0, 24)), np.zeros((0, 24))),
        (np.zeros(24), np.zeros(24)),
        (np.full((2, 24), np.nan), np.zeros((2, 24))),
        (np.zeros((2, 24)), np.full((2, 24), np.inf)),
    ],
)
def test_evaluation_rejects_invalid_inputs(predictions, targets):
    with pytest.raises(ValueError):
        updated_evaluate(
            StubModel(predictions),
            np.zeros((2, 2)),
            targets,
        )


def test_evaluation_rejects_overflowing_errors():
    predictions = np.full((2, 24), 1e308)
    targets = np.full((2, 24), -1e308)

    with pytest.raises(ValueError, match="too large"):
        updated_evaluate(
            StubModel(predictions),
            np.zeros((2, 2)),
            targets,
        )


def test_gait_validation_counts_violations_without_clipping():
    midpoint = (LOWER_LIMITS + UPPER_LIMITS) / 2
    gait = np.tile(midpoint, (300, 1))

    # Exact joint limits are allowed.
    gait[0] = LOWER_LIMITS
    gait[-1] = UPPER_LIMITS
    assert validate_gait(gait)["valid"]

    gait[4, 0] = LOWER_LIMITS[0] - 0.25
    gait[4, 1] = UPPER_LIMITS[1] + 0.5
    gait[12, 2] = UPPER_LIMITS[2] + 0.1
    original = gait.copy()

    result = validate_gait(gait)

    assert not result["valid"]
    assert result["out_of_bounds_count"] == 3
    assert result["invalid_frame_count"] == 2
    assert result["max_limit_violation_rad"] == pytest.approx(0.5)

    np.testing.assert_array_equal(gait, original)


@pytest.mark.parametrize(
    "gait",
    [
        np.zeros((299, 24)),
        np.zeros((300, 23)),
        np.full((300, 24), np.nan),
        np.full((300, 24), np.inf),
    ],
)
def test_gait_validation_rejects_invalid_arrays(gait):
    with pytest.raises(ValueError):
        validate_gait(gait)


def test_constant_gait_has_zero_cycle_steps():
    metrics = cycle_metrics(np.zeros((300, 24)))

    assert metrics["max_adjacent_step_rad"] == 0.0
    assert metrics["max_last_to_first_step_rad"] == 0.0


def test_cycle_metrics_include_last_to_first_transition():
    gait = np.zeros((300, 24))
    gait[:, 0] = np.linspace(0.0, 1.0, 300)

    metrics = cycle_metrics(gait)

    # Internal steps are small, but the cycle boundary jumps from 1 to 0.
    assert metrics["max_adjacent_step_rad"] == pytest.approx(1.0)
    assert metrics["max_last_to_first_step_rad"] == pytest.approx(1.0)


def test_cycle_metrics_detect_internal_jump():
    gait = np.zeros((300, 24))
    gait[100, 5] = 0.4

    metrics = cycle_metrics(gait)

    assert metrics["max_adjacent_step_rad"] == pytest.approx(0.4)
    assert metrics["max_last_to_first_step_rad"] == 0.0


@pytest.mark.parametrize(
    "gait",
    [
        np.zeros((299, 24)),
        np.zeros((300, 23)),
        np.zeros((0, 24)),
        np.full((300, 24), np.nan),
    ],
)
def test_cycle_metrics_reject_invalid_arrays(gait):
    with pytest.raises(ValueError):
        cycle_metrics(gait)
