
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

    
    def predict(self, X):
        """Generate predictions for a batch of inputs."""
        return self.forward(X)

    def train_batch(self, X, y, learning_rate):
        """Train one mini-batch using manual backpropagation."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2 or X.shape[1] != self.layer_sizes[0]:
            raise ValueError(
                f"X must have shape (batch_size, {self.layer_sizes[0]})"
            )
        if y.shape != (X.shape[0], self.layer_sizes[-1]):
            raise ValueError(
                f"y must have shape (batch_size, {self.layer_sizes[-1]})"
            )
        if X.shape[0] == 0:
            raise ValueError("Batch cannot be empty")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("Inputs and targets must be finite")
        if learning_rate <= 0 or not np.isfinite(learning_rate):
            raise ValueError("Learning rate must be positive and finite")

        # Forward propagation: cache activations and pre-activations.
        activations = [X]
        pre_activations = []
        activation = X

        for layer in range(len(self.weights)):
            z = activation @ self.weights[layer] + self.biases[layer]
            pre_activations.append(z)

            if layer < len(self.weights) - 1:
                activation = np.maximum(0, z)
            else:
                activation = z

            activations.append(activation)

        predictions = activations[-1]
        loss = self.mse_loss(predictions, y)

        # Derivative of mean squared error.
        delta = 2.0 * (predictions - y) / y.size

        grad_weights = [None] * len(self.weights)
        grad_biases = [None] * len(self.biases)

        # Backpropagation through all layers.
        for layer in reversed(range(len(self.weights))):
            grad_weights[layer] = activations[layer].T @ delta
            grad_biases[layer] = np.sum(delta, axis=0)

            if layer > 0:
                delta = delta @ self.weights[layer].T
                delta *= (pre_activations[layer - 1] > 0)

        # Gradient descent parameter updates.
        for layer in range(len(self.weights)):
            self.weights[layer] -= learning_rate * grad_weights[layer]
            self.biases[layer] -= learning_rate * grad_biases[layer]

        return loss

