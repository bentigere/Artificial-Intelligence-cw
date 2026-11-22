
import numpy as np


class NeuralNetwork:
    """Feedforward neural network for predicting spider joint angles."""

    def __init__(self, layer_sizes=(2, 64, 128, 64, 24), seed=42):
        self.layer_sizes = tuple(layer_sizes)

        if len(self.layer_sizes) < 2 or any(
            size <= 0 for size in self.layer_sizes
        ):
            raise ValueError("Invalid network architecture")

        rng = np.random.default_rng(seed)

        self.weights = []
        self.biases = []

        for input_size, output_size in zip(
            self.layer_sizes[:-1],
            self.layer_sizes[1:]
        ):
            # He initialization for ReLU hidden layers
            weight = rng.standard_normal(
                (input_size, output_size)
            ) * np.sqrt(2.0 / input_size)

            bias = np.zeros((1, output_size))

            self.weights.append(weight)
            self.biases.append(bias)

    @staticmethod
    def relu(x):
        return np.maximum(0.0, x)

    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)

    def forward(self, X):
        """Return predicted joint angles."""
        activation = np.asarray(X, dtype=float)

        if activation.ndim == 1:
            activation = activation.reshape(1, -1)

        if activation.ndim != 2 or activation.shape[1] != self.layer_sizes[0]:
            raise ValueError("Incorrect input dimensions")

        # ReLU hidden layers
        for weight, bias in zip(
            self.weights[:-1],
            self.biases[:-1]
        ):
            activation = self.relu(
                activation @ weight + bias
            )

        # Linear output layer for joint angles
        output = (
            activation @ self.weights[-1]
            + self.biases[-1]
        )

        return output

    @staticmethod
    def mse_loss(predictions, targets):
        predictions = np.asarray(predictions, dtype=float)
        targets = np.asarray(targets, dtype=float)

        if predictions.shape != targets.shape:
            raise ValueError("Prediction and target shapes differ")

        return float(np.mean((predictions - targets) ** 2))
