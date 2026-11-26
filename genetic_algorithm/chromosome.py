
"""Chromosome representation for the Genetic Algorithm.

A chromosome represents one complete spider walking cycle.
The spider has eight legs, each containing three joints, so
each frame contains 24 joint angles.

A complete gait consists of 300 frames, giving a chromosome
of 7,200 genes. Each gene represents a joint angle in radians.

This module creates random chromosomes within the robot's
joint limits and converts encoded chromosomes back into
frame-by-frame data for fitness evaluation and visualisation.
"""
import numpy as np

# Number of legs, joints, and frames
NUM_LEGS = 8
JOINTS_PER_LEG = 3
NUM_FRAMES = 300

# Total joint angles in one frame
ANGLES_PER_FRAME = NUM_LEGS * JOINTS_PER_LEG

# Total genes in one chromosome
NUM_GENES = NUM_FRAMES * ANGLES_PER_FRAME

# Order specified in the coursework
LEG_ORDER = [
    "L1", "L2", "L3", "L4",
    "R4", "R3", "R2", "R1"
]

JOINT_ORDER = [
    "hip_joint",
    "thigh_joint",
    "knee_joint"
]

# Joint limits from the Webots robot
# Each set of three values represents hip, thigh, knee
LOWER_LIMITS = np.tile(
    [-1.5708, -1.5708, -2.0],
    NUM_LEGS
)

UPPER_LIMITS = np.tile(
    [1.5708, 1.5708, 1.0],
    NUM_LEGS
)


def create_chromosome(rng=None):
   
    """Generate a random chromosome representing a complete spider gait.

    Each of the 300 frames contains 24 joint angles for the eight
    legs. Angles are generated within the permitted joint limits,
    ensuring that the initial population contains valid joint
    configurations.

    The chromosome is flattened into a one-dimensional array
    so the GA can apply selection, crossover and mutation.

    Args:
        rng: Optional NumPy random number generator, allowing
            reproducible chromosome generation.

    Returns:
        numpy.ndarray: A one-dimensional chromosome containing
        7,200 joint angles in radians.
    """

    if rng is None:
        rng = np.random.default_rng()

    gait = rng.uniform(
        low=LOWER_LIMITS,
        high=UPPER_LIMITS,
        size=(NUM_FRAMES, ANGLES_PER_FRAME)
    )

    chromosome = gait.flatten()

    return chromosome


def chromosome_to_gait(chromosome):

    """Convert an encoded chromosome into a frame-by-frame gait.

    The Genetic Algorithm stores each walking cycle as a flat
    array of genes. This function restores the original gait
    structure so that other parts of the project can access
    the joint angles for each frame.

    Args:
        chromosome: A one-dimensional array containing 7,200
            joint angles.

    Returns:
        numpy.ndarray: A two-dimensional array with shape
        (300, 24), representing the full walking cycle.

    Raises:
        ValueError: If the chromosome does not contain exactly
            7,200 genes.
    """

    chromosome = np.asarray(chromosome)

    if chromosome.size != NUM_GENES:
        raise ValueError(
            f"Expected {NUM_GENES} genes"
        )

    return chromosome.reshape(
        NUM_FRAMES,
        ANGLES_PER_FRAME
    )


# Test our chromosome encoding
if __name__ == "__main__":

    rng = np.random.default_rng(42)

    chromosome = create_chromosome(rng)

    print("Total genes:", len(chromosome))

    gait = chromosome_to_gait(chromosome)

    print("Gait shape:", gait.shape)

    print("Angles in first frame:")
    print(np.round(gait[0], 3))

    print("Number of frames:", len(gait))

    print("Angles per frame:", len(gait[0]))
