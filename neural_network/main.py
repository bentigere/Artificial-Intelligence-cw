"""Train and evaluate a fresh NumPy model in its own run directory."""

import argparse
import json
from pathlib import Path
from tempfile import mkdtemp

from neural_network.dataset import load_datasets
from neural_network.export_final import GA_TARGET, evaluate_cycle, export_gait
from neural_network.network import NeuralNetwork
from neural_network.train import EPOCHS, SEED, save_history, save_model, to_arrays, train
from neural_network.visualisation import create_evaluation_plots


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--output-dir", type=Path, help="A new directory for this run")
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("--epochs must be positive")

    if args.output_dir is None:
        runs = Path("runs/nn")
        runs.mkdir(parents=True, exist_ok=True)
        output_dir = Path(mkdtemp(prefix="demo-", dir=runs))
    else:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True)

    train_data, val_data, _ = load_datasets(GA_TARGET)
    X_train, y_train = to_arrays(train_data)
    X_val, y_val = to_arrays(val_data)
    model = NeuralNetwork(seed=SEED)
    history = train(model, X_train, y_train, X_val, y_val, epochs=args.epochs)
    save_model(model, output_dir / "model.npz")
    save_history(history, output_dir / "history.csv")

    predictions, targets, metrics = evaluate_cycle(model, GA_TARGET)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    create_evaluation_plots(predictions, targets, history, output_dir / "plots")

    for split in ("train", "validation", "test"):
        result = metrics[split]
        print(f"{split}: MSE={result['mse']:.6f}, MAE={result['mae']:.6f}")
    print(f"Model, metrics and plots saved to {output_dir}")
    export_gait(predictions, output_dir / "predicted_gait.csv")


if __name__ == "__main__":
    main()
