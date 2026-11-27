"""Pair gait phases with GA joint angles for training and evaluation."""

import csv
import math
import random
import numpy as np

from genetic_algorithm.chromosome import (
    LOWER_LIMITS,
    UPPER_LIMITS,
)


def load_datasets(file_path="data/best_gait_matlab.csv"):
    """Shuffle frames with a fixed seed, then split them 70/15/15."""
    with open(file_path, "r") as file:
        reader = csv.reader(file)

        gait = []

        for row in reader:
            if not row:
                continue
            angles = [float(value) for value in row]
            gait.append(angles)

    if len(gait) != 300:
        raise ValueError("Expected 300 frames")

    for frame in gait:
        if len(frame) != 24:
            raise ValueError("Each frame must contain 24 angles")

    gait_array = np.asarray(gait, dtype=float)

    if not np.isfinite(gait_array).all():
        raise ValueError("Gait contains NaN or infinite values")

    if (
        np.any(gait_array < LOWER_LIMITS)
        or np.any(gait_array > UPPER_LIMITS)
    ):
        raise ValueError("Gait contains angles outside joint limits")

    dataset = []

    for i in range(300):
        t = i / 300

        # Encode phase on a circle so it wraps smoothly at the cycle boundary.
        inputs = [math.sin(2 * math.pi * t), math.cos(2 * math.pi * t)]

        targets = gait[i]

        dataset.append((inputs, targets))

    rng = random.Random(42)
    rng.shuffle(dataset)

    train_size = int(0.70 * len(dataset))
    val_size = int(0.15 * len(dataset))

    train_data = dataset[:train_size]

    val_data = dataset[train_size : train_size + val_size]

    test_data = dataset[train_size + val_size :]

    return train_data, val_data, test_data
