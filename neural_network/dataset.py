"""
divides datasets into train, validation, and test
- 70%: train
- 15%: validation
- 15%: test
"""

import csv
import math
import random
import numpy as np

from genetic_algorithm.chromosome import (
    LOWER_LIMITS,
    UPPER_LIMITS,
)

def load_datasets(file_path="data/best_gait_matlab.csv"):
    # Read the gait data from the CSV file
    with open(file_path, "r") as file:
        reader = csv.reader(file)

        gait = []

        for row in reader:
            if not row:
                continue
            angles = [float(value) for value in row]
            gait.append(angles)

    # Check the dataset dimensions
    if len(gait) != 300:
        raise ValueError("Expected 300 frames")

    for frame in gait:
        if len(frame) != 24:
            raise ValueError("Each frame must contain 24 angles")

# Validate numerical values and joint limits
    gait_array = np.asarray(gait, dtype=float)

    if not np.isfinite(gait_array).all():
        raise ValueError("Gait contains NaN or infinite values")

    if (
    np.any(gait_array < LOWER_LIMITS)
    or np.any(gait_array > UPPER_LIMITS)
    ):
        raise ValueError("Gait contains angles outside joint limits")

    # Create input-output pairs
    dataset = []

    for i in range(300):
        # Position in walking cycle
        t = i / 300

        # Input features
        # Use sine and cosine to represent the spider's position in its
        # walking cycle, so the start and end of the cycle connect smoothly.
        inputs = [math.sin(2 * math.pi * t), math.cos(2 * math.pi * t)]

        # Target: 24 joint angles
        targets = gait[i]

        dataset.append((inputs, targets))

    # Shuffle the data reproducibly
    rng = random.Random(42)
    rng.shuffle(dataset)

    # Calculate split sizes
    train_size = int(0.70 * len(dataset))
    val_size = int(0.15 * len(dataset))

    # Split the data
    train_data = dataset[:train_size]

    val_data = dataset[train_size : train_size + val_size]

    test_data = dataset[train_size + val_size :]

    return train_data, val_data, test_data
