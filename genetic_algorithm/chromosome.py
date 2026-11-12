
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
    """
    Create one complete spider gait.

    Each frame contains 24 joint angles.
    There are 300 frames in total.

    The chromosome is stored as a flat array
    containing 7,200 values.

    All angles are in radians.
    """

    if rng is None:
        rng = np.random.default_rng()

    # Randomly generate 300 frames of 24 angles
    gait = rng.uniform(
        low=LOWER_LIMITS,
        high=UPPER_LIMITS,
        size=(NUM_FRAMES, ANGLES_PER_FRAME)
    )

    # Convert the 300 x 24 array into 7,200 genes
    chromosome = gait.flatten()

    return chromosome


def chromosome_to_gait(chromosome):
    """
    Convert a flat chromosome back into 300 frames.

    Input: 7,200 joint angles
    Output: array with shape (300, 24)
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
