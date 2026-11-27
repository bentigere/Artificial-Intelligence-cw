"""Compare the team's NumPy backpropagation with PyTorch autograd.

Run from repository root:
    python -m neural_network.pytorch_comparison
This creates comparison evidence only; it does not overwrite the final NN.
"""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np

try:
    import torch
    from torch import nn
except ImportError as exc:
    raise SystemExit("PyTorch is required: python -m pip install torch") from exc

from neural_network.dataset import load_datasets
from neural_network.network import NeuralNetwork
from neural_network.train import load_model, to_arrays


class TorchGaitNN(nn.Module):
    """Match the NumPy network's layers and activations."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(2, 64, dtype=torch.float64),
            nn.Linear(64, 128, dtype=torch.float64),
            nn.Linear(128, 64, dtype=torch.float64),
            nn.Linear(64, 24, dtype=torch.float64),
        ])

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return self.layers[-1](x)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("data/pytorch_comparison.json"))
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("Epochs, batch size, and learning rate must be positive")

    torch.set_num_threads(1)
    train_data, val_data, test_data = load_datasets("data/best_gait_matlab.csv")
    x_train, y_train = map(torch.from_numpy, to_arrays(train_data))
    x_val, y_val = map(torch.from_numpy, to_arrays(val_data))
    x_test_np, y_test_np = to_arrays(test_data)
    x_test = torch.from_numpy(x_test_np)

    model = TorchGaitNN()
    initial_numpy = NeuralNetwork(seed=42)

    # Match the NumPy model's initial weights for a controlled comparison.
    with torch.no_grad():
        for layer, weight, bias in zip(model.layers, initial_numpy.weights, initial_numpy.biases):
            layer.weight.copy_(torch.from_numpy(weight.T))
            layer.bias.copy_(torch.from_numpy(bias.ravel()))

    # Use the same SGD learning rate, minibatches, shuffling, and validation rule.
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    shuffle_rng = random.Random(42)
    best_val = float("inf")
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        indices = list(range(len(x_train)))
        shuffle_rng.shuffle(indices)
        model.train()
        for start in range(0, len(indices), args.batch_size):
            batch = indices[start:start + args.batch_size]
            optimizer.zero_grad()
            loss = torch.mean((model(x_train[batch]) - y_train[batch]) ** 2)
            loss.backward()  # PyTorch autograd replaces manual chain-rule code.
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_mse = torch.mean((model(x_train) - y_train) ** 2).item()
            val_mse = torch.mean((model(x_val) - y_val) ** 2).item()
        history.append((epoch, train_mse, val_mse))
        if val_mse < best_val:
            best_val = val_mse
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    with torch.no_grad():
        torch_prediction = model(x_test).numpy()

    # Both models are scored using exactly the same held-out examples.
    numpy_model = load_model("data/nn_final_model.npz")
    numpy_prediction = numpy_model.predict(x_test_np)

    def metrics(prediction):
        error = prediction - y_test_np
        return {"test_mse_rad2": float(np.mean(error ** 2)),
                "test_mae_rad": float(np.mean(np.abs(error)))}

    max_parameter_difference = max(
        max(np.max(np.abs(layer.weight.detach().numpy().T - weight)),
            np.max(np.abs(layer.bias.detach().numpy() - bias.ravel())))
        for layer, weight, bias in zip(model.layers, numpy_model.weights, numpy_model.biases)
    )

    result = {
        "method": "Matched architecture, initial weights, data split, SGD, minibatches and seed; PyTorch autograd vs NumPy manual backpropagation",
        "pytorch_version": torch.__version__,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "numpy_final": metrics(numpy_prediction),
        "pytorch": metrics(torch_prediction),
        "max_absolute_test_prediction_difference": float(np.max(np.abs(torch_prediction - numpy_prediction))),
        "max_absolute_parameter_difference": float(max_parameter_difference),
        "pytorch_best_validation_mse_rad2": float(best_val),
        "pytorch_final_training_mse_rad2": float(history[-1][1]),
        "pytorch_final_validation_mse_rad2": float(history[-1][2]),
        "note": "The NumPy checkpoint is from 2000 epochs. Use --epochs 2000 for an equal-training comparison. Test data are not used for training or checkpoint selection.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    history_path = args.output.with_name(args.output.stem + "_history.csv")
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_mse", "validation_mse"])
        writer.writerows(history)
    print(json.dumps(result, indent=2))
    print(f"Saved: {args.output} and {history_path}")


if __name__ == "__main__":
    main()