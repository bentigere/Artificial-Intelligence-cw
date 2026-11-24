import json
from pathlib import Path

import numpy as np

from neural_network.dataset import load_datasets
from neural_network.network import NeuralNetwork
from neural_network.train import (
    train,
    to_arrays,
    save_model,
    save_history,
)
from neural_network.updated_evaluate import (
    updated_evaluate,
    phase_inputs,
    validate_gait,
    cycle_metrics,
)
from neural_network.visualisation import create_evaluation_plots


def main():
    project_root = Path(__file__).resolve().parent.parent

    dataset_path = project_root / "data/best_gait_matlab.csv"
    model_path = project_root / "models/spider_nn.npz"
    history_path = project_root / "data/nn_training_history.csv"
    metrics_path = project_root / "data/nn_evaluation_metrics.json"
    prediction_path = project_root / "data/nn_predicted_gait.csv"
    plots_dir = project_root / "images/nn"

    # Training settings
    layer_sizes = (2, 64, 128, 64, 24)
    learning_rate = 0.001
    batch_size = 32
    epochs = 500
    seed = 42

    # Prepare data.
    train_data, val_data, test_data = load_datasets(dataset_path)

    X_train, y_train = to_arrays(train_data)
    X_val, y_val = to_arrays(val_data)
    X_test, y_test = to_arrays(test_data)

    model = NeuralNetwork(layer_sizes=layer_sizes, seed=seed)

    # Train and restore the best validation model.
    print("Training...")

    history = train(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        seed=seed,
    )

    save_model(model, model_path)
    save_history(history, history_path)

    # Evaluate each split separately.
    metrics = {
        "train": updated_evaluate(model, X_train, y_train),
        "validation": updated_evaluate(model, X_val, y_val),
        "test": updated_evaluate(model, X_test, y_test),
    }

    for name, result in metrics.items():
        print(f"{name}: " f"MSE={result['mse']:.6f}, " f"MAE={result['mae']:.6f}")

    # Predict the full gait in its original frame order.
    X_all = phase_inputs()
    y_all = np.loadtxt(dataset_path, delimiter=",")
    predictions = model.predict(X_all)

    metrics["all"] = updated_evaluate(model, X_all, y_all)
    metrics["joint_validation"] = validate_gait(predictions)
    metrics["prediction_cycle"] = cycle_metrics(predictions)
    metrics["target_cycle"] = cycle_metrics(y_all)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    create_evaluation_plots(
        predictions,
        y_all,
        history,
        plots_dir,
    )

    # Export for MATLAB only when all predicted angles are valid.
    if not metrics["joint_validation"]["valid"]:
        raise ValueError(
            "Predictions violate joint limits. "
            "Metrics and plots were saved; gait CSV was not exported."
        )

    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        prediction_path,
        predictions,
        delimiter=",",
        fmt="%.17g",
    )

    print(f"Model saved to: {model_path}")
    print(f"Predicted gait saved to: {prediction_path}")


if __name__ == "__main__":
    main()
