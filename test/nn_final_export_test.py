from pathlib import Path

import numpy as np
import pytest

from neural_network.export_final import (
    FINAL_MODEL,
    GA_TARGET,
    evaluate_cycle,
    export_gait,
)
from neural_network.train import load_model
from neural_network.updated_evaluate import validate_gait


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_final_checkpoint_exports_the_submitted_gait(tmp_path):
    model = load_model(PROJECT_ROOT / FINAL_MODEL)
    predictions, _, metrics = evaluate_cycle(model, PROJECT_ROOT / GA_TARGET)
    output = tmp_path / "gait.csv"
    export_gait(predictions, output)

    exported = np.loadtxt(output, delimiter=",")
    submitted = np.loadtxt(PROJECT_ROOT / "data/nn_predicted_gait.csv", delimiter=",")
    assert validate_gait(exported)["valid"]
    np.testing.assert_allclose(exported, submitted, atol=1e-10, rtol=0)
    assert metrics["test"]["sample_count"] == 45
    assert metrics["test"]["mse"] == pytest.approx(0.005115634728259382)


def test_invalid_gait_leaves_previous_csv_intact(tmp_path):
    output = tmp_path / "gait.csv"
    output.write_text("previous export\n", encoding="ascii")
    predictions = np.zeros((300, 24))
    predictions[0, 0] = 2

    with pytest.raises(ValueError, match="1 out-of-range angles"):
        export_gait(predictions, output)

    assert output.read_text(encoding="ascii") == "previous export\n"
