"""Training and checkpoint tests using small synthetic datasets."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from neural_network.network import NeuralNetwork
from neural_network.train import (
    load_model,
    save_history,
    save_model,
    to_arrays,
    train,
)


class RecordingNetwork:
    """Record training batches without changing predictions."""

    def __init__(self, prediction=0.0):
        self.batches = []
        self.prediction = prediction

    def train_batch(self, X, y, learning_rate):
        self.batches.append((X.copy(), y.copy(), learning_rate))
        return 0.0

    def predict(self, X):
        return np.full((len(X), 24), self.prediction)


class TestTraining(unittest.TestCase):
    def setUp(self):
        self.X = np.arange(14, dtype=float).reshape(7, 2) / 14
        self.y = np.repeat(self.X[:, :1], 24, axis=1)
        # Distinct validation examples make training-data leakage detectable.
        self.X_val = self.X[:2] + 10
        self.y_val = np.full((2, 24), 2.0)

    def test_to_arrays_preserves_pairs(self):
        data = [(self.X[i].tolist(), self.y[i].tolist()) for i in range(7)]
        X, y = to_arrays(data)

        self.assertEqual(X.shape, (7, 2))
        self.assertEqual(y.shape, (7, 24))
        np.testing.assert_array_equal(X, self.X)
        np.testing.assert_array_equal(y, self.y)

    def test_batches_cover_training_data_and_preserve_pairs(self):
        model = RecordingNetwork()
        train(
            model,
            self.X,
            self.y,
            self.X_val,
            self.y_val,
            epochs=2,
            batch_size=3,
            learning_rate=0.01,
        )

        self.assertEqual(len(model.batches), 6)
        for epoch in range(2):
            batches = model.batches[epoch * 3 : (epoch + 1) * 3]
            self.assertEqual([len(X) for X, _, _ in batches], [3, 3, 1])
            seen = []
            for X, y, learning_rate in batches:
                self.assertEqual(learning_rate, 0.01)
                for inputs, targets in zip(X, y):
                    matches = np.flatnonzero(np.all(self.X == inputs, axis=1))
                    self.assertEqual(len(matches), 1)
                    index = int(matches[0])
                    np.testing.assert_array_equal(targets, self.y[index])
                    seen.append(index)
            self.assertEqual(sorted(seen), list(range(7)))

    def test_shuffle_is_reproducible(self):
        models = [RecordingNetwork(), RecordingNetwork()]
        histories = []
        for model in models:
            histories.append(
                train(
                    model,
                    self.X,
                    self.y,
                    self.X_val,
                    self.y_val,
                    epochs=3,
                    batch_size=3,
                    seed=17,
                )
            )

        np.testing.assert_array_equal(histories[0], histories[1])
        for first, second in zip(models[0].batches, models[1].batches):
            np.testing.assert_array_equal(first[0], second[0])
            np.testing.assert_array_equal(first[1], second[1])

    def test_history_records_training_and_validation_mse(self):
        history = train(
            RecordingNetwork(),
            self.X,
            self.y,
            self.X_val,
            self.y_val,
            epochs=3,
            batch_size=3,
        )

        self.assertEqual(history.shape, (3, 3))
        np.testing.assert_array_equal(history[:, 0], [1, 2, 3])
        np.testing.assert_allclose(history[:, 1], np.mean(self.y**2))
        np.testing.assert_allclose(history[:, 2], 4.0)

    def test_non_finite_loss_is_rejected(self):
        for prediction in (np.nan, np.inf):
            with self.subTest(prediction=prediction):
                with self.assertRaisesRegex(ValueError, "Non-finite loss at epoch 1"):
                    train(
                        RecordingNetwork(prediction),
                        self.X,
                        self.y,
                        self.X_val,
                        self.y_val,
                        epochs=1,
                    )

    def test_real_network_training_reduces_loss(self):
        model = NeuralNetwork(layer_sizes=(2, 24), seed=7)
        initial_loss = np.mean((model.predict(self.X) - self.y) ** 2)
        history = train(
            model,
            self.X,
            self.y,
            self.X[:2],
            self.y[:2],
            epochs=80,
            batch_size=3,
            learning_rate=0.1,
        )

        self.assertTrue(np.isfinite(history).all())
        self.assertLess(history[-1, 1], initial_loss)
        np.testing.assert_allclose(
            history[-1, 1], np.mean((model.predict(self.X) - self.y) ** 2)
        )
        np.testing.assert_allclose(
            history[-1, 2],
            np.mean((model.predict(self.X[:2]) - self.y[:2]) ** 2),
        )

    def test_save_history_creates_directory_and_writes_csv(self):
        history = np.array([[1, 0.123456789, 0.25], [2, 0.1, 0.2]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data" / "history.csv"
            save_history(history, path)

            self.assertEqual(
                path.read_text().splitlines()[0], "epoch,train_mse,val_mse"
            )
            loaded = np.loadtxt(path, delimiter=",", skiprows=1)
            np.testing.assert_allclose(loaded, history, rtol=1e-9)

    def test_checkpoint_restores_trained_parameters_and_predictions(self):
        for layer_sizes in ((2, 24), (2, 8, 24), (2, 8, 6, 24)):
            with self.subTest(layer_sizes=layer_sizes):
                model = NeuralNetwork(layer_sizes=layer_sizes, seed=7)
                train(
                    model,
                    self.X,
                    self.y,
                    self.X_val,
                    self.y_val,
                    epochs=2,
                    batch_size=3,
                )
                expected = model.predict(self.X)

                with tempfile.TemporaryDirectory() as directory:
                    # A custom suffix must not cause an extra .npz suffix.
                    path = Path(directory) / "models" / "checkpoint.bin"
                    save_model(model, path)
                    self.assertTrue(path.is_file())
                    loaded = load_model(path)

                self.assertEqual(loaded.layer_sizes, layer_sizes)
                np.testing.assert_array_equal(loaded.predict(self.X), expected)
                for i in range(len(model.weights)):
                    np.testing.assert_array_equal(loaded.weights[i], model.weights[i])
                    np.testing.assert_array_equal(loaded.biases[i], model.biases[i])
                loaded.weights[0][0, 0] += 1
                np.testing.assert_array_equal(model.predict(self.X), expected)

    def test_load_rejects_incorrect_parameter_shapes(self):
        for parameter, wrong_shape, message in (
            ("weight_0", (3, 24), "Incorrect weight shape at layer 0"),
            ("bias_0", (24,), "Incorrect bias shape at layer 0"),
        ):
            with self.subTest(parameter=parameter):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "invalid.npz"
                    parameters = {
                        "layer_sizes": np.array([2, 24]),
                        "weight_0": np.zeros((2, 24)),
                        "bias_0": np.zeros((1, 24)),
                    }
                    parameters[parameter] = np.zeros(wrong_shape)
                    np.savez(path, **parameters)

                    with self.assertRaisesRegex(ValueError, message):
                        load_model(path)


if __name__ == "__main__":
    unittest.main()
