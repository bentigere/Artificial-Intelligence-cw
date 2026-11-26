from pathlib import Path
import numpy as np
import random
from neural_network.network import NeuralNetwork

# config

LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 500
SEED = 42

# Converts the gait dataset into NumPy arrays for neural network training.
def to_arrays(data):
    """Convert (inputs, targets) pairs into NumPy arrays."""
    X = np.asarray([inputs for inputs, _ in data], dtype=float)
    y = np.asarray([targets for _, targets in data], dtype=float)
    return X, y

# Trains the neural network using mini-batch gradient descent and backpropagation.
# Validation loss is monitored to identify the best-performing model.
def train(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    learning_rate=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    seed=SEED,
):
    rng = random.Random(seed)
    history = []

    best_val_loss = float("inf")
    best_weights = None
    best_biases = None

    for epoch in range(1, epochs + 1):
        # Shuffle indices to keep inputs and targets paired.
        indices = list(range(len(X_train)))
        rng.shuffle(indices)

        # Mini-batch training balances efficient weight updates with stable learning.
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]

            # A small learning rate allows gradual weight updates and reduces instability.
            model.train_batch(
                X_train[batch],
                y_train[batch],
                learning_rate,
            )

        # Measure both losses using the model after this epoch.
        train_loss = np.mean((model.predict(X_train) - y_train) ** 2)
        val_loss = np.mean((model.predict(X_val) - y_val) ** 2)

        if not np.isfinite(train_loss) or not np.isfinite(val_loss):
            raise ValueError(f"Non-finite loss at epoch {epoch}")

        history.append((epoch, train_loss, val_loss))
        
        # Restore the weights with the lowest validation loss rather than
        # using the final epoch, which may perform worse.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = [weight.copy() for weight in model.weights]
            best_biases = [bias.copy() for bias in model.biases]

    if best_weights is not None:
        model.weights = best_weights
        model.biases = best_biases

    return np.asarray(history)

# Saves the training and validation losses for analysing convergence.
def save_history(history, path="data/nn_training_history.csv"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savetxt(
        path,
        history,
        delimiter=",",
        header="epoch,train_mse,val_mse",
        comments="",
        fmt=["%d", "%.10g", "%.10g"],
    )

# Saves the trained weights and biases so the model can be reused.
def save_model(model, path="models/spider_nn.npz"):
    """Save the architecture, weights and biases."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    parameters = {
        "layer_sizes": np.asarray(model.layer_sizes, dtype=int),
    }

    for i in range(len(model.weights)):
        parameters[f"weight_{i}"] = model.weights[i]
        parameters[f"bias_{i}"] = model.biases[i]

    # Opening the file explicitly preserves the supplied filename.
    with path.open("wb") as file:
        np.savez(file, **parameters)

# Loads the saved weights and biases to restore the trained network.
def load_model(path="models/spider_nn.npz"):
    """Rebuild the network and restore its parameters."""
    with np.load(path, allow_pickle=False) as parameters:
        layer_sizes = tuple(int(size) for size in parameters["layer_sizes"])
        model = NeuralNetwork(layer_sizes=layer_sizes)

        for i in range(len(model.weights)):
            weight = parameters[f"weight_{i}"]
            bias = parameters[f"bias_{i}"]

            if weight.shape != model.weights[i].shape:
                raise ValueError(f"Incorrect weight shape at layer {i}")

            if bias.shape != model.biases[i].shape:
                raise ValueError(f"Incorrect bias shape at layer {i}")

            model.weights[i] = weight.copy()
            model.biases[i] = bias.copy()

    return model
