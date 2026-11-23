[course_work_description](../../../Downloads/M33174_cw_item1_.pdf)

[forward_leg_kinematics2](forward_leg_kinematics2.m)
returns 4 coordinates for 1 leg
j1: the attachment point of the leg to the body.
j2: the connection between the coxa and femur.
j3: the connection between the femur and tibia.
j4: the foot tip.

## run GA

1. every time run GA, change the output csv file path

```terminal
python test/ga.py
```

## animate spider model

1. run the main.m file through Matlab
   - change the csv file path

## Neural network evaluation (Member 4)

Run from the repository root with Python 3.12+ and NumPy/Matplotlib installed:

```sh
python -m pip install numpy matplotlib pytest
python -m neural_network.evaluate
python -m pytest -q
```

The evaluator loads `data/nn_reference_model.npz`, compares all ordered phases with
`data/best_gait_matlab.csv`, and exports **`data/nn_predicted_gait.csv` (300 rows × 24 columns)**.
It also saves overall and train/validation/test MSE/MAE, per-joint errors and limit
diagnostics to `data/nn_evaluation_metrics.json`, plus the three plots in `images/nn/`.
Invalid predictions fail rather than being clipped or exported.

No saved model was present on main when evaluation was added. The committed checkpoint
is a labelled reference using the team's existing trainer defaults. To reproduce it:

```sh
python -m neural_network.reference_training
python -m neural_network.evaluate
```

For another saved model/history, use `--model PATH --history PATH`. Checkpoints use
NumPy NPZ arrays named `layer_sizes`, `weight_0`, `bias_0`, etc.; `save_model()` in
`neural_network/evaluate.py` exports a trained `NeuralNetwork` without pickle. The
phase feature and output column conventions must match those above.

For MATLAB, run **`main_nn`** from this repository to play the predicted gait and save
five pose snapshots. This reuses the lecturer's functions and keeps the GA viewer intact.
MATLAB execution remains unverified here because no runtime is installed and MATLAB
Online needs sign-in. See [evaluation and comparison](neural_network/evaluation_report.md)
for measured results, the weak reference-model fit, and limitations.
