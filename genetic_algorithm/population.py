
import numpy as np

from genetic_algorithm.chromosome import (
    chromosome_to_gait,
    NUM_FRAMES,
    ANGLES_PER_FRAME,
    LOWER_LIMITS,
    UPPER_LIMITS,
)

POPULATION_SIZE = 20

# Webots standing angles
RIGHT_POSE = [0.0, 1.07, -0.58]
LEFT_POSE = [0.0, -1.07, 0.58]

# MATLAB reference angles
MATLAB_REFERENCE_POSE = [0.0, -np.pi / 4, -np.pi / 3]

# Required leg order: L1 L2 L3 L4 R4 R3 R2 R1
STANDING_POSE = np.array(LEFT_POSE * 4 + RIGHT_POSE * 4)
MATLAB_STANDING_POSE = np.tile(MATLAB_REFERENCE_POSE, 8)

# Alternating leg phases
LEG_PHASES = [0, np.pi] * 4


def create_smooth_chromosome(rng, reference_pose=STANDING_POSE):
    """Create a smooth 300-frame gait with 7200 genes."""

    time = np.linspace(
        0, 2 * np.pi, NUM_FRAMES, endpoint=False
    )

    reference_pose = np.asarray(reference_pose, dtype=float)

    if reference_pose.shape != (ANGLES_PER_FRAME,):
        raise ValueError("Reference pose must contain 24 angles")

    # Start every frame at the standing position
    gait = np.tile(MATLAB_STANDING_POSE, (NUM_FRAMES, 1))


    # Add smooth periodic joint movements
    for leg in range(8):
        phase = LEG_PHASES[leg]
        start = leg * 3

        for joint in range(3):
            amplitude = rng.uniform(0.03, 0.10)

            gait[:, start + joint] += (
                amplitude * np.sin(time + phase)
            )

    # Enforce joint limits
    gait = np.clip(
        gait,
        LOWER_LIMITS + 0.001,
        UPPER_LIMITS - 0.001
    )

    return gait.flatten()


def create_population(
    size=POPULATION_SIZE,
    reference_pose=STANDING_POSE
):
    """Create smooth candidate gaits for MATLAB or Webots."""

    rng = np.random.default_rng(42)

    population = []

    for _ in range(size):
        chromosome = create_smooth_chromosome(
            rng,
            reference_pose=reference_pose
        )

        population.append(chromosome)

    return np.array(population)


if __name__ == "__main__":

    # Test MATLAB population initialization
    population = create_population(
        reference_pose=MATLAB_STANDING_POSE
    )

    first_gait = chromosome_to_gait(population[0])

    print("Population shape:", population.shape)
    print("First gait shape:", first_gait.shape)
    print("First frame:", first_gait[0].reshape(8, 3))

    changes = np.diff(first_gait, axis=0)

    print(
        "Maximum joint change:",
        np.max(np.abs(changes))
    )

    print(
        "All angles within limits:",
        bool(
            np.all(first_gait >= LOWER_LIMITS)
            and np.all(first_gait <= UPPER_LIMITS)
        )
    )
