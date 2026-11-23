
import unittest
import numpy as np

from neural_network.network import NeuralNetwork


class TestNeuralNetwork(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.nn = NeuralNetwork()

    def test_output_shape(self):
        X = np.random.randn(32, 2)
        output = self.nn.predict(X)
        self.assertEqual(output.shape, (32, 24))

    def test_weight_shapes(self):
        expected = [
            (2, 64),
            (64, 128),
            (128, 64),
            (64, 24),
        ]
        actual = [w.shape for w in self.nn.weights]
        self.assertEqual(actual, expected)

    def test_loss_is_zero_for_identical_values(self):
        y = np.random.randn(10, 24)
        loss = self.nn.mse_loss(y, y)
        self.assertAlmostEqual(loss, 0.0)

    def test_training_reduces_loss(self):
        X = np.random.randn(32, 2)
        y = np.random.randn(32, 24)

        initial = self.nn.mse_loss(
            self.nn.predict(X), y
        )

        for _ in range(200):
            self.nn.train_batch(X, y, 0.001)

        final = self.nn.mse_loss(
            self.nn.predict(X), y
        )

        self.assertLess(final, initial)


if __name__ == "__main__":
    unittest.main()
