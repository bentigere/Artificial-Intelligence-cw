"""Evaluate the saved final NumPy model and export one cycle for MATLAB."""

import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from neural_network.dataset import load_datasets
from neural_network.train import load_model, to_arrays
from neural_network.updated_evaluate import (
    calculate_metrics,
    cycle_metrics,
    phase_inputs,
    updated_evaluate,
    validate_gait,
)

FINAL_MODEL = Path("data/nn_final_model.npz")
GA_TARGET = Path("data/best_gait_matlab.csv")
FINAL_OUTPUT = Path("data/nn_predicted_gait.csv")


def evaluate_cycle(model, dataset_path):
    """Compare the model with the GA data by split and over the ordered cycle."""
    splits = [to_arrays(split) for split in load_datasets(dataset_path)]
    metrics = {
        name: updated_evaluate(model, inputs, targets)
        for name, (inputs, targets) in zip(("train", "validation", "test"), splits, strict=True)
    }

    _, training_targets = splits[0]
    _, test_targets = splits[2]
    baseline = np.broadcast_to(training_targets.mean(axis=0), test_targets.shape)
    baseline_errors = calculate_metrics(baseline, test_targets)
    metrics["training_mean_baseline"] = {
        "mse": baseline_errors["mse"],
        "mae": baseline_errors["mae"],
    }

    targets = np.loadtxt(dataset_path, delimiter=",")
    predictions = model.predict(phase_inputs())
    metrics["all"] = calculate_metrics(predictions, targets)
    metrics["joint_validation"] = validate_gait(predictions)
    metrics["prediction_cycle"] = cycle_metrics(predictions)
    metrics["target_cycle"] = cycle_metrics(targets)
    return predictions, targets, metrics


def export_gait(predictions, output):
    """Replace the CSV only after validation and a successful write."""
    validation = validate_gait(predictions)
    if not validation["valid"]:
        raise ValueError(f"Gait has {validation['out_of_bounds_count']} out-of-range angles")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = NamedTemporaryFile(
        mode="w", encoding="ascii", newline="\n", dir=output.parent, delete=False
    )
    staged_path = Path(staging.name)
    try:
        with staging:
            np.savetxt(staging, predictions, delimiter=",", fmt="%.10f")
        staged_path.replace(output)
    finally:
        staged_path.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=FINAL_OUTPUT)
    parser.add_argument("--metrics", type=Path, help="Write the detailed metrics as JSON")
    args = parser.parse_args(argv)

    protected = {FINAL_MODEL.resolve(), GA_TARGET.resolve()}
    outputs = {args.output.resolve()}
    if args.metrics is not None:
        outputs.add(args.metrics.resolve())
        if len(outputs) != 2:
            parser.error("the CSV and metrics need distinct paths")
    if protected & outputs:
        parser.error("output paths must differ from the model and target paths")

    predictions, _, metrics = evaluate_cycle(load_model(FINAL_MODEL), GA_TARGET)
    export_gait(predictions, args.output)

    if args.metrics is not None:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(
            json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    print(f"Test MSE {metrics['test']['mse']:.8f} rad²")
    print(f"Test MAE {metrics['test']['mae']:.8f} rad")
    print(f"Exported {predictions.shape[0]} frames × {predictions.shape[1]} angles to {args.output}")


if __name__ == "__main__":
    main()
