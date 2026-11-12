import numpy as np

from genetic_algorithm.chromosome import (
    chromosome_to_gait,
    NUM_FRAMES,
    ANGLES_PER_FRAME,
    LOWER_LIMITS,
    UPPER_LIMITS,
)

POPULATION_SIZE = 20

# Standing angles from the Webots controller
# Each leg has: hip, thigh, knee
RIGHT_POSE = [0.0, 1.07, -0.58]
LEFT_POSE = [0.0, -1.07, 0.58]

# MATLAB reference pose, shared by all eight legs
# coxa: 0, femur: -45, tibia: -60
MATLAB_REFERENCE_POSE = [0.0, -np.pi / 4, -np.pi / 3]

# Required leg order: L1 L2 L3 L4 R4 R3 R2 R1
STANDING_POSE = np.array(LEFT_POSE * 4 + RIGHT_POSE * 4)

# 24 angles in the required leg order
MATLAB_STANDING_POSE = np.tile(MATLAB_REFERENCE_POSE, 8)

# Alternate the leg movement phases
LEG_PHASES = [0, np.pi] * 4


def create_smooth_chromosome(rng):
    """
    Create a smooth 300-frame walking candidate.

    The spider starts near its standing pose.
    The joint angles vary smoothly using sine waves.

    Output: one chromosome containing 7200 genes.
    """

    # Time values covering one walking cycle
    time = np.linspace(0, 2 * np.pi, NUM_FRAMES, endpoint=False)

    # Start every frame at the standing position
    gait = np.tile(MATLAB_STANDING_POSE, (NUM_FRAMES, 1))

    # Add small periodic movements to each leg
    for leg in range(8):

        phase = LEG_PHASES[leg]
        start = leg * 3

        for joint in range(3):

            # Small random movement amplitude
            amplitude = rng.uniform(0.03, 0.10)

            # Smooth movement around standing angle
            gait[:, start + joint] += amplitude * np.sin(time + phase)

    # Keep all angles within the motor limits
    gait = np.clip(gait, LOWER_LIMITS + 0.001, UPPER_LIMITS - 0.001)

    # Convert the 300 x 24 gait into 7200 genes
    return gait.flatten()


def create_population(size=POPULATION_SIZE):
    """Generate a population of smooth candidate gaits."""

    rng = np.random.default_rng(42)

    population = []

    for _ in range(size):
        chromosome = create_smooth_chromosome(rng)
        population.append(chromosome)

    return np.array(population)


# Test the initialization
if __name__ == "__main__":

    population = create_population()

    print("Population size:", len(population))
    print("Population shape:", population.shape)

    first_gait = chromosome_to_gait(population[0])

    print("First gait shape:", first_gait.shape)

    # Measure the largest change between consecutive frames
    changes = np.diff(first_gait, axis=0)

    print("Maximum joint change between frames:", np.max(np.abs(changes)))

    print(
        "All angles within limits:",
        bool(np.all(first_gait >= LOWER_LIMITS) and np.all(first_gait <= UPPER_LIMITS)),
    )

    print(
        "First two chromosomes identical:", np.array_equal(population[0], population[1])
    )
