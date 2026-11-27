"""Store a 300-frame gait as 7,200 joint angles in radians."""

import numpy as np

NUM_LEGS = 8
JOINTS_PER_LEG = 3
NUM_FRAMES = 300
ANGLES_PER_FRAME = NUM_LEGS * JOINTS_PER_LEG
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

# Webots joint limits in radians, in hip, thigh, knee order.
LOWER_LIMITS = np.tile(
    [-1.5708, -1.5708, -2.0],
    NUM_LEGS
)

UPPER_LIMITS = np.tile(
    [1.5708, 1.5708, 1.0],
    NUM_LEGS
)


def create_chromosome(rng=None):
    """Sample a flat gait within joint limits, optionally with a supplied RNG."""
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
    """Reshape 7,200 angles into 300 frames of 24 angles each."""
    chromosome = np.asarray(chromosome)

    if chromosome.size != NUM_GENES:
        raise ValueError(
            f"Expected {NUM_GENES} genes"
        )

    return chromosome.reshape(
        NUM_FRAMES,
        ANGLES_PER_FRAME
    )


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
