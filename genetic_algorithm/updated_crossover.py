import random
import numpy as np

from genetic_algorithm.chromosome import chromosome_to_gait

"""
The original crossover chooses each of the 7,200 angles independently from either parent. This can disrupt smooth movement and coordination between joints.

The updated crossover chooses one parent per leg and copies that leg’s three joint angles across all 300 frames together. This preserves each leg’s complete movement while combining legs from different parents.
"""


def updated_crossover(parent1, parent2):
    """Inherit each leg's complete movement from one parent."""
    gait1 = chromosome_to_gait(parent1)
    gait2 = chromosome_to_gait(parent2)

    child = np.empty_like(gait1)

    for leg in range(8):
        start = leg * 3
        end = start + 3

        parent = gait1 if random.random() < 0.5 else gait2

        # Copy all frames and all three joints for this leg.
        child[:, start:end] = parent[:, start:end]

    return child.flatten().tolist()
