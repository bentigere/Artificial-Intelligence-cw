"""Save neural-network evaluation figures without an interactive display.

All joint arrays use radians and the Part 1 leg/joint order. Predictions are
plotted as supplied: clipping them here would hide invalid model outputs.
"""

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from genetic_algorithm.chromosome import (
    ANGLES_PER_FRAME,
    JOINT_ORDER,
    LEG_ORDER,
    NUM_FRAMES,
)


def _validate_inputs(predictions, targets, history):
    predictions = np.asarray(predictions, dtype=float)
    targets = np.asarray(targets, dtype=float)
    expected_shape = (NUM_FRAMES, ANGLES_PER_FRAME)

    for name, values in (("Predictions", predictions), ("Targets", targets)):
        if values.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must contain only finite joint angles")

    history = np.asarray(history, dtype=float)
    if history.ndim != 2 or history.shape[1] != 3 or history.shape[0] == 0:
        raise ValueError(
            "History must have shape (epochs, 3): epoch, train_mse, val_mse"
        )
    if not np.isfinite(history).all():
        raise ValueError("History must contain only finite values")

    epochs = history[:, 0]
    if (
        np.any(epochs <= 0)
        or np.any(epochs != np.floor(epochs))
        or np.any(np.diff(epochs) <= 0)
    ):
        raise ValueError("History epochs must be positive, increasing integers")
    if np.any(history[:, 1:] < 0):
        raise ValueError("Training and validation MSE must be non-negative")

    return predictions, targets, history


def _figure(figsize):
    figure = Figure(figsize=figsize, layout="constrained")
    FigureCanvasAgg(figure)
    return figure


def _save(figure, path):
    figure.savefig(path, dpi=150, facecolor="white")
    figure.clear()


def create_evaluation_plots(predictions, targets, history, output_dir):
    """Save prediction, joint error and loss figures and return their paths.

    ``predictions`` and ``targets`` must each be finite 300-by-24 arrays in
    ``LEG_ORDER``/``JOINT_ORDER``. ``history`` has one row per recorded epoch
    with columns ``epoch, train_mse, val_mse``.
    """
    predictions, targets, history = _validate_inputs(predictions, targets, history)
    with np.errstate(over="ignore", invalid="ignore"):
        errors = predictions - targets
        mse_by_joint = np.mean(errors**2, axis=0)
        mae_by_joint = np.mean(np.abs(errors), axis=0)
    if not np.isfinite(mse_by_joint).all() or not np.isfinite(mae_by_joint).all():
        raise ValueError("Joint errors exceed the supported numeric range")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "joint_predictions": output_dir / "nn_joint_predictions.png",
        "joint_errors": output_dir / "nn_joint_errors.png",
        "training_loss": output_dir / "nn_training_loss.png",
    }

    figure = _figure((13, 20))
    axes = figure.subplots(len(LEG_ORDER), len(JOINT_ORDER), sharex=True)
    phases = np.arange(NUM_FRAMES) / NUM_FRAMES
    joint_names = [name.removesuffix("_joint").capitalize() for name in JOINT_ORDER]
    for leg_index, leg in enumerate(LEG_ORDER):
        for joint_index, joint in enumerate(joint_names):
            axis = axes[leg_index, joint_index]
            column = leg_index * len(JOINT_ORDER) + joint_index
            axis.plot(
                phases,
                targets[:, column],
                color="#2166ac",
                linewidth=1.5,
                linestyle="--",
                label="GA target",
            )
            axis.plot(
                phases,
                predictions[:, column],
                color="#d95f02",
                linewidth=1.5,
                label="NN prediction",
            )
            axis.set_title(f"{leg} · {joint}", fontsize=11)
            axis.grid(alpha=0.25)
            axis.set_xlim(0, 1)
            if joint_index == 0:
                axis.set_ylabel("Angle (rad)")
            if leg_index == len(LEG_ORDER) - 1:
                axis.set_xlabel("Gait phase (cycle fraction)")
    axes[0, 0].legend(fontsize=8, loc="best")
    figure.suptitle(
        "Predicted versus GA joint angles · all 300 gait phases", fontsize=17
    )
    _save(figure, paths["joint_predictions"])

    labels = [f"{leg} {joint.lower()}" for leg in LEG_ORDER for joint in joint_names]
    figure = _figure((14, 8))
    axes = figure.subplots(2, 1, sharex=True)
    for axis, values, label, color in (
        (axes[0], mse_by_joint, "MSE (rad²)", "#2166ac"),
        (axes[1], mae_by_joint, "MAE (rad)", "#d95f02"),
    ):
        axis.bar(np.arange(ANGLES_PER_FRAME), values, color=color, width=0.75)
        axis.set_ylabel(label)
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    axes[1].set_xticks(np.arange(ANGLES_PER_FRAME), labels, rotation=60, ha="right")
    axes[1].set_xlabel("Joint in Part 1 column order")
    figure.suptitle("Prediction error by joint · all 300 gait phases", fontsize=17)
    _save(figure, paths["joint_errors"])

    figure = _figure((10, 5))
    axis = figure.subplots()
    axis.plot(history[:, 0], history[:, 1], color="#2166ac", label="Training MSE")
    axis.plot(history[:, 0], history[:, 2], color="#d95f02", label="Validation MSE")
    axis.set(xlabel="Epoch", ylabel="MSE (rad²)", title="Training and validation loss")
    axis.set_ylim(bottom=0)
    axis.legend()
    axis.grid(alpha=0.25)
    _save(figure, paths["training_loss"])

    return paths
