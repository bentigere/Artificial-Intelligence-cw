"""Reproduce a reference checkpoint for evaluation using the team's trainer.

This is an evaluation reference, not a claim that the team has selected its
final model. The network, split and backpropagation implementations are reused
without modification; test samples are not used in training.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import numpy as np

from neural_network.dataset import load_datasets
from neural_network.network import NeuralNetwork

from neural_network.train import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    SEED,
    save_history,
    save_model,
    to_arrays,
    train,
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_main_commit():
    """Record the checkout's main baseline when Git is available."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "origin/main"],
            cwd=Path(__file__).resolve().parent.parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=Path("data/best_gait_matlab.csv"))
    parser.add_argument("--model", type=Path, default=Path("data/nn_reference_model.npz"))
    parser.add_argument("--history", type=Path, default=Path("data/nn_training_history.csv"))
    parser.add_argument("--provenance", type=Path, default=Path("data/nn_reference_training.json"))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.seed < 0:
        parser.error("epochs and batch size must be positive; seed must be non-negative")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning rate must be positive and finite")
    outputs = {args.model.resolve(), args.history.resolve(), args.provenance.resolve()}
    if len(outputs) != 3 or args.target.resolve() in outputs:
        parser.error("output paths must be distinct and must not replace the target gait")

    train_data, validation_data, _ = load_datasets(file_path=args.target)
    X_train, y_train = to_arrays(train_data)
    X_val, y_val = to_arrays(validation_data)
    model = NeuralNetwork(seed=args.seed)
    history = train(
        model, X_train, y_train, X_val, y_val,
        learning_rate=args.learning_rate, batch_size=args.batch_size,
        epochs=args.epochs, seed=args.seed,
    )
    save_model(model, args.model)
    save_history(history, args.history)

    package_dir = Path(__file__).resolve().parent
    provenance = {
        "label": "Reference checkpoint trained for evaluation; not the team's final selected model.",
        "seed": args.seed,
        "dataset_split_seed": 42,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "architecture": list(model.layer_sizes),
        "target_path": str(args.target),
        "target_sha256": _sha256(args.target),
        "model_path": str(args.model),
        "model_sha256": _sha256(args.model),
        "history_path": str(args.history),
        "history_sha256": _sha256(args.history),
        "source_main_commit": _source_main_commit(),
        "source_sha256": {name: _sha256(package_dir / name)
                          for name in ("network.py", "dataset.py", "train.py")},
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "train_sample_count": len(train_data),
        "validation_sample_count": len(validation_data),
        "first_epoch_train_mse": float(history[0, 1]),
        "first_epoch_validation_mse": float(history[0, 2]),
        "last_epoch_train_mse": float(history[-1, 1]),
        "last_epoch_validation_mse": float(history[-1, 2]),
    }
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved evaluation reference checkpoint: {args.model}")
    print(f"Final training MSE: {history[-1, 1]:.10g}; validation MSE: {history[-1, 2]:.10g} rad^2")


if __name__ == "__main__":
    main()
