import random

import numpy as np

from genetic_algorithm.chromosome import chromosome_to_gait

def updated_crossover(parent1, parent2):

    """
    Combine parent gaits while preserving smooth movement.
    Each parent contains 7200 genes (300 frames x 24 angles).
    The same blending ratio is used across all legs and
    joints to preserve the parents' coordinated gait patterns.
    """

    gait1 = chromosome_to_gait(parent1)
    gait2 = chromosome_to_gait(parent2)

    if gait1.shape != (300, 24) or gait2.shape != (300, 24):
        raise ValueError("Both parents must have shape (300, 24)")
    
    # Choose a random contribution from each parent
    alpha = random.uniform(0.25, 0.75)

    # Blend entire gait trajectories
    child = (
        alpha * gait1
        + (1.0 - alpha) * gait2
    )

    # Return the original chromosome structure
    return child.flatten().tolist()