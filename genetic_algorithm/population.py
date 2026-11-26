
"""Population management for the spider gait Genetic Algorithm.

A population is a collection of candidate spider walking gaits.
Each chromosome represents a complete walking cycle containing
300 frames and 24 joint angles per frame.

The initial population provides different candidate movements
for the Genetic Algorithm to evaluate. The fitness function
penalises movement errors, allowing selection and genetic
operators to favour better walking patterns over generations.

This module supports the creation and management of candidate
solutions during the optimisation process.
"""

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

# Required spider leg order
LEGS = ["L1", "L2", "L3", "L4", "R4", "R3", "R2", "R1"]

# Two alternating groups of four legs
GROUP_A = {"L1", "L3", "R4", "R2"}
GROUP_B = {"L2", "L4", "R3", "R1"}

# Opposite movement phases
LEG_PHASES = [0.0 if leg in GROUP_A else np.pi for leg in LEGS]


def create_smooth_chromosome(rng, reference_pose=STANDING_POSE):
    """Creates a random, smooth walking gait around the reference pose."""
    reference_pose = np.asarray(reference_pose, dtype=float)

    if reference_pose.shape != (ANGLES_PER_FRAME,):
        raise ValueError("Expected 24 reference joint angles")

    # One complete periodic gait cycle
    time = np.linspace(0, 2 * np.pi, NUM_FRAMES, endpoint=False)

    # Start from the chosen standing pose
    gait = np.tile(reference_pose, (NUM_FRAMES, 1))

    for leg in range(8):

        phase = LEG_PHASES[leg]
        start = leg * 3

        # Small variations between chromosomes
        hip_amplitude = rng.uniform(0.10, 0.20)
        thigh_amplitude = rng.uniform(0.08, 0.16)
        knee_amplitude = rng.uniform(0.06, 0.12)

        cycle = time + phase

        # Coxa: horizontal sweep
        hip_movement = hip_amplitude * np.cos(cycle)

        # Femur: candidate lifting motion
        thigh_movement = thigh_amplitude * np.sin(cycle)

        # Tibia: offset from femur motion
        knee_movement = knee_amplitude * np.sin(cycle - np.pi / 4)

        gait[:, start] += hip_movement
        gait[:, start + 1] += thigh_movement
        gait[:, start + 2] += knee_movement

    # Respect robot joint limits
    gait = np.clip(gait, LOWER_LIMITS + 0.001, UPPER_LIMITS - 0.001)

    return gait.flatten()


def create_population(size=POPULATION_SIZE, reference_pose=STANDING_POSE):
    """Creates the initial population of spider gaits for the GA."""
    rng = np.random.default_rng(42)

    population = []

    for _ in range(size):
        chromosome = create_smooth_chromosome(rng, reference_pose=reference_pose)

        population.append(chromosome)

    return np.array(population)


if __name__ == "__main__":

    # Test MATLAB population initialization
    population = create_population(reference_pose=MATLAB_STANDING_POSE)

    first_gait = chromosome_to_gait(population[0])

    print("Population shape:", population.shape)
    print("First gait shape:", first_gait.shape)
    print("First frame:", first_gait[0].reshape(8, 3))

    changes = np.diff(first_gait, axis=0)

    print("Maximum joint change:", np.max(np.abs(changes)))

    print(
        "All angles within limits:",
        bool(np.all(first_gait >= LOWER_LIMITS) and np.all(first_gait <= UPPER_LIMITS)),
    )
