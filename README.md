## Part 1: Genetic Algorithm

### See the spider animation

1. open the project folder in Matlab
2. run the `ga_main.m`

### Understanding the outputs

- `data/best_gait_matlab.csv` contains 300 frames of 24 joint
  angles in radians. Each row represents one spider pose.
- `data/fitness_history.csv` records the best and average fitness
  scores for each generation.
- Higher fitness scores are better. Scores are negative because
  the fitness function subtracts penalties.
- The fitness plot shows whether candidate solutions improve
  over generations.
- The MATLAB animation displays the generated gait with a fixed
  body; it does not simulate physical forward movement.

### Generate GA

Run from the project root

```bash
python -m genetic_algorithm.main
```

or if you use uv

```bash
uv run python -m genetic_algorithm.main
```

## Part 2: Neural Network

### see the spider animation

1. open the project folder in Matlab
2. run the `nn_main.m`

### Understanding the outputs

- `models/spider_nn.npz` stores the trained network.
- `data/nn_training_history.csv` records training progress.
- `data/nn_predicted_gait.csv` contains predicted joint angles,
  exported when the evaluation checks permit.

### Train the neural network

Run from the project root after generating the GA gait:

```bash
python -m neural_network.main
```

Alternatively, with uv:

```bash
uv run python -m neural_network.main
```
