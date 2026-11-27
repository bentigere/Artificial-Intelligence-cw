import random

import numpy as np

from genetic_algorithm.chromosome import chromosome_to_gait


def updated_crossover(parent1, parent2):
    """Blend complete parent gaits with one ratio to preserve smooth motion."""
    gait1 = chromosome_to_gait(parent1)
    gait2 = chromosome_to_gait(parent2)

    if gait1.shape != (300, 24) or gait2.shape != (300, 24):
        raise ValueError("Both parents must have shape (300, 24)")

    alpha = random.uniform(0.25, 0.75)

    child = alpha * gait1 + (1.0 - alpha) * gait2

    return child.flatten().tolist()
