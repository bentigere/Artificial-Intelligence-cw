import random
import numpy as np

from genetic_algorithm.chromosome import (
    chromosome_to_gait,
    LOWER_LIMITS,
    UPPER_LIMITS,
)


def updated_mutate(
    chromosome,
    mutation_rate=0.1,
    mutation_strength=0.05,
    bump_frames=60,
):
    """Add a smooth change across consecutive frames of a joint."""
    gait = chromosome_to_gait(chromosome).astype(float, copy=True)
    frames, joints = gait.shape

    if not 0 <= mutation_rate <= 1:
        raise ValueError("mutation_rate must be between 0 and 1.")
    if mutation_strength < 0:
        raise ValueError("mutation_strength must be non-negative.")
    if not 3 <= bump_frames <= frames:
        raise ValueError("Invalid number of bump frames.")

    # A smooth bump: zero at both ends, highest near the middle.
    bump = np.sin(np.linspace(0, np.pi, bump_frames)) ** 2
    bump[0] = bump[-1] = 0.0

    for joint in range(joints):
        if random.random() < mutation_rate:
            start = random.randrange(frames)
            amount = random.gauss(0, mutation_strength)

            # Continue at frame 0 if we pass the last frame.
            indices = (start + np.arange(bump_frames)) % frames

            gait[indices, joint] += amount * bump

    # Keep every joint within its bounds.
    gait = np.clip(gait, LOWER_LIMITS, UPPER_LIMITS)

    return gait.flatten().tolist()
