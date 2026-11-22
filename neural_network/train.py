from pathlib import Path
import numpy as np
from neural_network.dataset import load_datasets
import random

# config

LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 500
SEED = 42


def to_arrays(data):
    """Convert (inputs, targets) pairs into NumPy arrays."""
    X = np.asarray([inputs for inputs, _ in data], dtype=float)
    y = np.asarray([targets for _, targets in data], dtype=float)
    return X, y


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

    for epoch in range(1, epochs + 1):
        # Shuffle indices to keep inputs and targets paired.
        indices = list(range(len(X_train)))
        rng.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]

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

    return np.asarray(history)


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
