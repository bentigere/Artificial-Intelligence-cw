"""Evaluate a saved NumPy network against the ordered MATLAB GA gait.

Run from the repository root with ``python -m neural_network.evaluate``.
Errors are measured on raw predictions; invalid angles are never clipped.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from genetic_algorithm.chromosome import (
    ANGLES_PER_FRAME,
    JOINT_ORDER,
    LEG_ORDER,
    LOWER_LIMITS,
    NUM_FRAMES,
    UPPER_LIMITS,
)
from neural_network.dataset import load_datasets
from neural_network.network import NeuralNetwork
from neural_network.train import to_arrays
from neural_network.updated_evaluate import (
    _angles,
    calculate_metrics,
    cycle_metrics as _cycle_metrics,
    phase_inputs,
    validate_gait,
)
from neural_network.visualisation import create_evaluation_plots


def _checkpoint_arrays(model):
    layers = np.asarray(model.layer_sizes)
    if (
        layers.ndim != 1
        or layers.size < 2
        or layers.dtype.kind not in "iu"
        or np.any(layers <= 0)
        or layers[0] != 2
        or layers[-1] != 24
    ):
        raise ValueError("Evaluation requires a network with 2 inputs and 24 outputs")
    if len(model.weights) != layers.size - 1 or len(model.biases) != layers.size - 1:
        raise ValueError("Checkpoint layer counts do not match the architecture")
    arrays = {"layer_sizes": layers}
    for i, (input_size, output_size) in enumerate(zip(layers[:-1], layers[1:])):
        weight = np.asarray(model.weights[i], dtype=float)
        bias = np.asarray(model.biases[i], dtype=float)
        if weight.shape != (input_size, output_size) or bias.shape not in (
            (output_size,),
            (1, output_size),
        ):
            raise ValueError(f"Incorrect checkpoint parameter shape at layer {i}")
        if not np.isfinite(weight).all() or not np.isfinite(bias).all():
            raise ValueError("Checkpoint weights and biases must be finite")
        arrays[f"weight_{i}"] = weight
        arrays[f"bias_{i}"] = bias.reshape(1, -1)
    return arrays


def save_model(model, path):
    """Store exact trained weights in a portable NPZ archive, without pickle."""
    arrays = _checkpoint_arrays(model)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        np.savez_compressed(file, **arrays)


def load_model(path):
    """Load our NPZ contract: layer_sizes, weight_0, bias_0, and so on."""
    try:
        with np.load(path, allow_pickle=False) as checkpoint:
            layers = checkpoint["layer_sizes"]
            if (
                layers.ndim != 1
                or layers.size < 2
                or layers.dtype.kind not in "iu"
                or np.any(layers <= 0)
                or layers[0] != 2
                or layers[-1] != 24
            ):
                raise ValueError(
                    "Evaluation requires a network with 2 inputs and 24 outputs"
                )
            model = NeuralNetwork(layer_sizes=tuple(int(size) for size in layers))
            model.weights = [
                checkpoint[f"weight_{i}"].copy() for i in range(layers.size - 1)
            ]
            model.biases = [
                checkpoint[f"bias_{i}"].copy() for i in range(layers.size - 1)
            ]
        arrays = _checkpoint_arrays(model)
        model.biases = [arrays[f"bias_{i}"] for i in range(layers.size - 1)]
        return model
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Cannot load model checkpoint {path}: {exc}") from exc


def evaluate(
    model_path,
    target_path,
    history_path,
    output_path="data/nn_predicted_gait.csv",
    plots_dir="images/nn",
    metrics_path="data/nn_evaluation_metrics.json",
):
    inputs = {Path(path).resolve() for path in (model_path, target_path, history_path)}
    destinations = [Path(output_path), Path(metrics_path)] + [
        Path(plots_dir) / name
        for name in (
            "nn_joint_predictions.png",
            "nn_joint_errors.png",
            "nn_training_loss.png",
        )
    ]
    outputs = {path.resolve() for path in destinations}
    if len(outputs) != len(destinations) or inputs & outputs:
        raise ValueError(
            "Output paths must be distinct and must not replace evaluation inputs"
        )
    model = load_model(model_path)
    targets = np.loadtxt(target_path, delimiter=",")
    if not validate_gait(targets)["valid"]:
        raise ValueError("GA targets violate joint limits")
    predictions = _angles(model.predict(phase_inputs()))
    validity = validate_gait(predictions)
    if not validity["valid"]:
        raise ValueError(
            f"Predictions violate joint limits; no CSV exported: {validity}"
        )

    # Reuse Member 1's deterministic split, not a new shuffled evaluation split.
    splits = load_datasets(file_path=target_path)
    history = np.loadtxt(history_path, delimiter=",", skiprows=1, ndmin=2)
    metrics = {
        "all": calculate_metrics(predictions, targets),
        "joint_validation": validity,
        "units": {"mse": "rad^2", "mae": "rad"},
        "joint_order": [f"{leg}_{joint}" for leg in LEG_ORDER for joint in JOINT_ORDER],
        "phase_count": NUM_FRAMES,
        "prediction_cycle": _cycle_metrics(predictions),
        "target_cycle": _cycle_metrics(targets),
        "model_path": str(model_path),
        "model_sha256": hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
        "target_path": str(target_path),
        "target_sha256": hashlib.sha256(Path(target_path).read_bytes()).hexdigest(),
        "history_path": str(history_path),
        "history_sha256": hashlib.sha256(Path(history_path).read_bytes()).hexdigest(),
        "interpretation": "Random phase holdout from one gait: interpolation, not generalisation to new gaits or terrain.",
    }
    for name, samples in zip(("train", "validation", "test"), splits):
        inputs, target_subset = to_arrays(samples)
        metrics[name] = {
            "sample_count": len(samples),
            **calculate_metrics(model.predict(inputs), target_subset),
        }
    train_mean = np.mean(to_arrays(splits[0])[1], axis=0)
    test_targets = to_arrays(splits[2])[1]
    metrics["test_train_mean_baseline"] = calculate_metrics(
        np.tile(train_mean, (len(test_targets), 1)), test_targets
    )

    # Plotting validates the loss-history contract before the export is written.
    paths = create_evaluation_plots(predictions, targets, history, plots_dir)
    metrics["plots"] = {name: str(path) for name, path in paths.items()}
    output_path, metrics_path = Path(output_path), Path(metrics_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, predictions, delimiter=",", fmt="%.17g")
    metrics_path.write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="data/nn_reference_model.npz")
    parser.add_argument("--target", default="data/best_gait_matlab.csv")
    parser.add_argument("--history", default="data/nn_training_history.csv")
    parser.add_argument("--output", default="data/nn_predicted_gait.csv")
    parser.add_argument("--plots-dir", default="images/nn")
    parser.add_argument("--metrics", default="data/nn_evaluation_metrics.json")
    args = parser.parse_args(argv)
    try:
        metrics = evaluate(
            args.model,
            args.target,
            args.history,
            args.output,
            args.plots_dir,
            args.metrics,
        )
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Evaluation failed: {exc}\n")
    for split in ("all", "train", "validation", "test"):
        print(
            f"{split}: MSE={metrics[split]['mse']:.8g} rad^2; MAE={metrics[split]['mae']:.8g} rad"
        )
    print(f"Saved valid 300 x 24 gait to {args.output}")


if __name__ == "__main__":
    main()
