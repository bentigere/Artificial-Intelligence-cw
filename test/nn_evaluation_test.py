"""Contract tests for saved-model evaluation and the MATLAB gait export."""

import json

import numpy as np
import pytest

from genetic_algorithm.chromosome import LOWER_LIMITS, UPPER_LIMITS
from neural_network.evaluate import (
    calculate_metrics,
    evaluate,
    load_model,
    phase_inputs,
    save_model,
    validate_gait,
)
from neural_network.network import NeuralNetwork


def constant_model(angles):
    model = NeuralNetwork(layer_sizes=(2, 24), seed=9)
    model.weights[0][:] = 0.0
    model.biases[0][:] = angles
    return model


def write_case(tmp_path, angles=None):
    if angles is None:
        angles = (LOWER_LIMITS + UPPER_LIMITS) / 2.0
    model_path = tmp_path / "trained_model.npz"
    target_path = tmp_path / "targets.csv"
    history_path = tmp_path / "history.csv"
    save_model(constant_model(angles), model_path)
    np.savetxt(target_path, np.tile(angles, (300, 1)), delimiter=",")
    np.savetxt(
        history_path,
        [[1, 0.4, 0.5], [2, 0.1, 0.2]],
        delimiter=",",
        header="epoch,train_mse,val_mse",
        comments="",
    )
    return model_path, target_path, history_path


def test_phase_inputs_are_ordered_periodic_features():
    inputs = phase_inputs()
    assert inputs.shape == (300, 2)
    assert np.isfinite(inputs).all()
    phases = 2.0 * np.pi * np.arange(300) / 300
    np.testing.assert_allclose(inputs[:, 0], np.sin(phases), atol=1e-14)
    np.testing.assert_allclose(inputs[:, 1], np.cos(phases), atol=1e-14)
    np.testing.assert_allclose(inputs[0], [0.0, 1.0], atol=1e-14)
    assert not np.array_equal(inputs[-1], inputs[0])


def test_metrics_match_hand_calculation_per_joint():
    targets = np.zeros((2, 24))
    predictions = np.zeros_like(targets)
    predictions[:, 0] = [1.0, -3.0]
    metrics = calculate_metrics(predictions, targets)
    assert metrics["mse"] == pytest.approx(5.0 / 24)
    assert metrics["mae"] == pytest.approx(2.0 / 24)
    np.testing.assert_allclose(metrics["per_joint_mse"], [5.0] + [0.0] * 23)
    np.testing.assert_allclose(metrics["per_joint_mae"], [2.0] + [0.0] * 23)


@pytest.mark.parametrize(
    "predictions,targets",
    [
        (np.zeros((2, 23)), np.zeros((2, 23))),
        (np.zeros((2, 24)), np.zeros((3, 24))),
        (np.zeros((0, 24)), np.zeros((0, 24))),
        (np.zeros(24), np.zeros(24)),
        (np.full((2, 24), np.nan), np.zeros((2, 24))),
        (np.zeros((2, 24)), np.full((2, 24), np.inf)),
    ],
)
def test_metrics_reject_invalid_inputs(predictions, targets):
    with pytest.raises(ValueError):
        calculate_metrics(predictions, targets)


def test_gait_validation_counts_raw_violations_without_clipping():
    gait = np.tile((LOWER_LIMITS + UPPER_LIMITS) / 2.0, (300, 1))
    gait[0] = LOWER_LIMITS
    gait[-1] = UPPER_LIMITS
    assert validate_gait(gait)["valid"]
    gait[4, 0] = LOWER_LIMITS[0] - 0.25
    gait[4, 1] = UPPER_LIMITS[1] + 0.5
    gait[12, 2] = UPPER_LIMITS[2] + 0.1
    before = gait.copy()
    validation = validate_gait(gait)
    assert not validation["valid"]
    assert validation["out_of_bounds_count"] == 3
    assert validation["invalid_frame_count"] == 2
    assert validation["max_limit_violation_rad"] == pytest.approx(0.5)
    np.testing.assert_array_equal(gait, before)


@pytest.mark.parametrize(
    "gait",
    [np.zeros((299, 24)), np.zeros((300, 23)), np.full((300, 24), np.nan)],
)
def test_gait_validation_rejects_bad_shape_and_nonfinite_values(gait):
    with pytest.raises(ValueError):
        validate_gait(gait)


def test_saved_model_roundtrip_preserves_predictions(tmp_path):
    model = NeuralNetwork(layer_sizes=(2, 7, 24), seed=19)
    model_path = tmp_path / "checkpoint.npz"
    save_model(model, model_path)
    loaded = load_model(model_path)
    assert tuple(loaded.layer_sizes) == (2, 7, 24)
    np.testing.assert_array_equal(loaded.predict(phase_inputs()), model.predict(phase_inputs()))


@pytest.mark.parametrize("layer_sizes", [(3, 24), (2, 23)])
def test_saved_model_requires_two_inputs_and_24_outputs(tmp_path, layer_sizes):
    with pytest.raises(ValueError):
        save_model(NeuralNetwork(layer_sizes=layer_sizes), tmp_path / "bad_model.npz")


def test_saved_model_rejects_nonfinite_weights(tmp_path):
    model = NeuralNetwork(layer_sizes=(2, 24))
    model.weights[0][0, 0] = np.nan
    with pytest.raises(ValueError):
        save_model(model, tmp_path / "nan_model.npz")


def test_saved_model_rejects_malformed_archive(tmp_path):
    model_path = tmp_path / "missing_weights.npz"
    np.savez(model_path, layer_sizes=np.array([2, 24]))
    with pytest.raises(ValueError):
        load_model(model_path)


@pytest.mark.parametrize("kind", ["wrong_shape", "nonfinite", "object_array"])
def test_saved_model_rejects_corrupt_weight_arrays(tmp_path, kind):
    model_path = tmp_path / "corrupt.npz"
    weight = np.zeros((2, 24))
    if kind == "wrong_shape":
        weight = np.zeros((3, 24))
    elif kind == "nonfinite":
        weight[0, 0] = np.inf
    else:
        weight = np.array([[object()]], dtype=object)
    np.savez(
        model_path,
        layer_sizes=np.array([2, 24]),
        weight_0=weight,
        bias_0=np.zeros((1, 24)),
    )
    with pytest.raises(ValueError):
        load_model(model_path)


def test_evaluate_exports_all_ordered_phases_and_reports_heldout_metrics(tmp_path):
    model_path, target_path, history_path = write_case(tmp_path)
    # Distinct joint trajectories detect phase or column reordering in the CSV.
    model = constant_model((LOWER_LIMITS + UPPER_LIMITS) / 2.0)
    model.weights[0][0] = np.linspace(0.01, 0.04, 24)
    model.weights[0][1] = np.linspace(-0.03, 0.02, 24)
    save_model(model, model_path)
    np.savetxt(target_path, model.predict(phase_inputs()), delimiter=",")
    output_path = tmp_path / "output" / "nn_predicted_gait.csv"
    metrics_path = tmp_path / "output" / "metrics.json"
    plots_dir = tmp_path / "plots"
    metrics = evaluate(
        model_path,
        target_path,
        history_path,
        output_path=output_path,
        plots_dir=plots_dir,
        metrics_path=metrics_path,
    )
    exported = np.loadtxt(output_path, delimiter=",")
    assert exported.shape == (300, 24)
    np.testing.assert_allclose(exported, np.loadtxt(target_path, delimiter=","))
    for split in ("all", "train", "validation", "test"):
        assert metrics[split]["mse"] == pytest.approx(0.0, abs=1e-20)
        assert metrics[split]["mae"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["joint_validation"]["valid"]
    assert [metrics[name]["sample_count"] for name in ("train", "validation", "test")] == [210, 45, 45]
    assert json.loads(metrics_path.read_text(encoding="utf-8"))["all"] == metrics["all"]
    expected = {"nn_joint_predictions.png", "nn_joint_errors.png", "nn_training_loss.png"}
    assert {p.name for p in plots_dir.glob("*.png")} == expected
    for path in plots_dir.glob("*.png"):
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_evaluate_does_not_replace_outputs_with_out_of_bounds_predictions(tmp_path):
    model_path, target_path, history_path = write_case(tmp_path)
    invalid = constant_model(UPPER_LIMITS + 0.3)
    save_model(invalid, model_path)
    output_path = tmp_path / "nn_predicted_gait.csv"
    output_path.write_text("previous valid output\n", encoding="utf-8")
    plots_dir = tmp_path / "plots"
    metrics_path = tmp_path / "metrics.json"
    with pytest.raises(ValueError):
        evaluate(
            model_path,
            target_path,
            history_path,
            output_path=output_path,
            plots_dir=plots_dir,
            metrics_path=metrics_path,
        )
    assert output_path.read_text(encoding="utf-8") == "previous valid output\n"
    assert not plots_dir.exists()
    assert not metrics_path.exists()


@pytest.mark.parametrize("alias", ["output_target", "metrics_output", "plot_target"])
def test_evaluate_rejects_aliased_destinations_without_changing_inputs(tmp_path, alias):
    model_path, target_path, history_path = write_case(tmp_path)
    output_path = tmp_path / "output" / "nn_predicted_gait.csv"
    metrics_path = tmp_path / "output" / "metrics.json"
    plots_dir = tmp_path / "plots"
    if alias == "output_target":
        output_path = target_path
    elif alias == "metrics_output":
        metrics_path = output_path
    else:
        plots_dir.mkdir()
        plot_target = plots_dir / "nn_training_loss.png"
        plot_target.write_bytes(target_path.read_bytes())
        target_path = plot_target

    original_inputs = {
        path: path.read_bytes() for path in (model_path, target_path, history_path)
    }
    original_files = {path for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="Output paths must be distinct"):
        evaluate(
            model_path,
            target_path,
            history_path,
            output_path=output_path,
            plots_dir=plots_dir,
            metrics_path=metrics_path,
        )

    assert {path for path in tmp_path.rglob("*") if path.is_file()} == original_files
    for path, contents in original_inputs.items():
        assert path.read_bytes() == contents
    assert not (tmp_path / "output").exists()
