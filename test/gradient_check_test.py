
import unittest
import numpy as np

from neural_network.network import NeuralNetwork


class TestGradientChecking(unittest.TestCase):

    def test_output_weight_gradient(self):
        nn = NeuralNetwork(
            layer_sizes=(2, 4, 3, 24),
            seed=42
        )

        X = np.array([
            [0.2, 0.8],
            [0.7, -0.3]
        ])

        y = np.full((2, 24), 0.5)

        epsilon = 1e-5
        learning_rate = 1e-4

        layer = -1
        row, col = 0, 0

        original = nn.weights[layer][row, col]

        nn.weights[layer][row, col] = original + epsilon
        loss_plus = nn.mse_loss(nn.predict(X), y)

        nn.weights[layer][row, col] = original - epsilon
        loss_minus = nn.mse_loss(nn.predict(X), y)

        nn.weights[layer][row, col] = original

        numerical_gradient = (
            loss_plus - loss_minus
        ) / (2 * epsilon)

        nn.train_batch(X, y, learning_rate)

        analytical_gradient = (
            original - nn.weights[layer][row, col]
        ) / learning_rate

        np.testing.assert_allclose(
            analytical_gradient,
            numerical_gradient,
            rtol=1e-4,
            atol=1e-6
        )

    
    def test_all_layer_gradients(self):
        nn = NeuralNetwork(
            layer_sizes=(2, 4, 3, 24),
            seed=42
        )

        X = np.array([
            [0.2, 0.8],
            [0.7, -0.3]
        ])
        y = np.full((2, 24), 0.5)

        epsilon = 1e-5
        learning_rate = 1e-4

        # Save parameters before training.
        original_weights = [w.copy() for w in nn.weights]
        original_biases = [b.copy() for b in nn.biases]

        # Calculate analytical gradients using one update.
        nn.train_batch(X, y, learning_rate)

        analytical_weights = [
            (old - new) / learning_rate
            for old, new in zip(original_weights, nn.weights)
        ]
        analytical_biases = [
            (old - new) / learning_rate
            for old, new in zip(original_biases, nn.biases)
        ]

        # Restore parameters before numerical checking.
        nn.weights = [w.copy() for w in original_weights]
        nn.biases = [b.copy() for b in original_biases]

        for layer in range(len(nn.weights)):
            for parameter, analytical in (
                (nn.weights[layer], analytical_weights[layer]),
                (nn.biases[layer], analytical_biases[layer])
            ):
                for index in np.ndindex(parameter.shape):
                    original = parameter[index]

                    parameter[index] = original + epsilon
                    loss_plus = nn.mse_loss(nn.predict(X), y)

                    parameter[index] = original - epsilon
                    loss_minus = nn.mse_loss(nn.predict(X), y)

                    parameter[index] = original

                    numerical = (
                        loss_plus - loss_minus
                    ) / (2 * epsilon)

                    np.testing.assert_allclose(
                        analytical[index],
                        numerical,
                        rtol=1e-4,
                        atol=1e-6
                    )



if __name__ == "__main__":
    unittest.main()
