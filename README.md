# Spider gait optimisation and prediction

M33174 Artificial Intelligence group coursework. Part 1 uses a genetic algorithm (GA) to find a 300-frame spider gait. Part 2 trains a neural network (NN) to predict that gait from the phase of the walking cycle.

## Set up and test

Use Python 3.11–3.14 and run the commands from the repository root. Create a virtual environment first.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
python -m pytest
```

On Windows PowerShell, activate the environment with `.\.venv\Scripts\Activate.ps1`. Install `".[test,comparison]"` if you also want to run the optional PyTorch comparison. Use MATLAB for the 3D playback.

## View the saved gaits

Each gait CSV has 300 rows and 24 columns with no header. Angles are in radians. Each frame contains hip, thigh and knee angles for each leg in this order.

```text
L1, L2, L3, L4, R4, R3, R2, R1
```

Set the MATLAB Current Folder to the repository root.

| MATLAB command | Gait |
| --- | --- |
| `ga_main` | GA gait from `data/best_gait_matlab.csv` |
| `nn_main` | NN gait selected by `nn_gait_type` in `nn_main.m` |

Set `nn_gait_type = "raw";` for `data/nn_predicted_gait.csv`, or `nn_gait_type = "smoothed";` for `data/nn_final_smoothed_gait.csv`. The default is smoothed. The viewers use the same spider geometry and keep the body fixed. They show how the legs move through the cycle. Forward travel and physical stability require a dynamics simulation.

Run `validate_smoothed_gait` in MATLAB to check the saved smoothed CSV. To recompute it, run `test_periodic_smoothing`, which applies an 11-frame circular moving average and writes `data/nn_final_smoothed_gait.csv`.

## Part 1 · Genetic algorithm

Each chromosome stores a 300-frame gait as 7,200 angles. The GA uses tournament selection, blends complete parent gaits with a shared weight, and mutates contiguous frames of a joint with a smooth offset. Elitism preserves the best gait between generations. Forward kinematics turn angles into foot positions. Fitness rewards following the support, swing and horizontal foot targets, avoiding ground penetration and moving smoothly.

| Setting | Value |
| --- | ---: |
| Population | 20 |
| Generations | 100 |
| Tournament size | 3 |
| Mutation rate | 0.1 |
| Mutation strength | 0.15 |
| Seed | 42 |

Best fitness rose from −150.0052 to −35.0511, and average valid fitness rose from −176.2865 to −36.2288. Higher scores indicate smaller movement penalties. [GA testing and figures](docs/integration_testing.md)

```bash
python -m genetic_algorithm.main
```

This run writes `data/best_gait_matlab.csv` and `data/fitness_history.csv`. Run it in a separate checkout or copy if you want to preserve the bundled GA target for the final NN experiment.

The optional Webots demonstration opens `Spider_world_demo/worlds/spider_world_1.wbt` and uses the gait CSV next to its `gait_player` controller. Its input is separate from the MATLAB GA and NN gaits.

## Part 2 · Neural network

The NumPy model takes the sine and cosine of the gait phase and predicts 24 angles. It uses layers of size 2 → 64 → 128 → 64 → 24, ReLU in the hidden layers and a linear output. Training uses mini-batch gradient descent, mean squared error (MSE) and manual backpropagation.

The final run used 2,000 epochs, learning rate 0.001, batch size 32 and seed 42. The 300 GA frames were split into 210 for training, 45 for validation and 45 for testing. The saved checkpoint has a held-out MSE of **0.00511563 rad²** and MAE of **0.05726265 rad**. These metrics describe unseen phases of the same GA gait. [Evaluation and comparison](neural_network/evaluation_report.md)

| File | Contents |
| --- | --- |
| [`data/nn_final_model.npz`](data/nn_final_model.npz) | Final saved checkpoint |
| [`data/nn_final_history.csv`](data/nn_final_history.csv) and [`data/nn_final_info.json`](data/nn_final_info.json) | Loss by epoch and training configuration |
| [`data/nn_final_metrics.json`](data/nn_final_metrics.json) | Final checkpoint evaluation |
| [`data/nn_predicted_gait.csv`](data/nn_predicted_gait.csv) | Final raw predictions for all 300 phases |
| [`data/nn_final_gait.csv`](data/nn_final_gait.csv) | Same raw joint angles, used by the MATLAB comparison scripts |
| [`data/nn_final_smoothed_gait.csv`](data/nn_final_smoothed_gait.csv) | Smoothed predictions for playback |
| [`images/nn_final/`](images/nn_final/) | Training and validation loss, GA versus NN joint angles, joint errors and the pose comparison |

### Evaluate the saved checkpoint

```bash
python -m neural_network.export_final
```

This evaluates the final checkpoint, validates the 24 joint angles per frame and refreshes `data/nn_predicted_gait.csv`. Set `--output PATH` to export to another CSV or `--metrics PATH` to save detailed metrics to JSON.

### Run training or the library comparison

```bash
python -m neural_network.main
python -m neural_network.reference_training --epochs 2000
python -m neural_network.pytorch_comparison --output runs/nn/pytorch_comparison.json
```

The first command runs the 500-epoch smoke test and saves its model, predictions, metrics and plots in a fresh `runs/nn/demo-*` folder. The second repeats the final 2,000-epoch training configuration and saves its checkpoint, history and metadata under `runs/nn/reference/`. Numerical reproduction depends on the software environment. The third runs the optional PyTorch comparison with the same architecture, weights at initialisation, split and training batches as the final NumPy run.
