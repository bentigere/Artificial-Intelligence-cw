# evaluate how realistic it is

# define the ground value
import numpy as np

from kinematics import forward_leg_kinematics2
from chromosome import chromosome_to_gait
from population import MATLAB_STANDING_POSE

LEG_NAMES = ["L1", "L2", "L3", "L4", "R4", "R3", "R2", "R1"]
BASE_ANGLES = np.deg2rad([45, 75, 105, 135, -135, -105, -75, -45])
SEGMENT_LENGTHS = [1.2, 0.7, 1.0]

# Black-bar intervals: percentages of the cycle spent supporting.
# start and end
# https://pmc.ncbi.nlm.nih.gov/articles/PMC6935789/ figure2
SUPPORT_INTERVALS = {
    "L1": [(0, 67)],
    "L2": [(0, 20), (51, 100)],
    "L3": [(14, 89)],
    "L4": [(0, 44), (75, 100)],
    "R1": [(0, 12), (39, 100)],
    "R2": [(0, 70), (93, 100)],
    "R3": [(0, 36), (71, 100)],
    "R4": [(28, 93)],
}


def get_foot_positions(pose):
    """Convert 24 joint angles into eight foot-tip coordinates."""
    pose = np.asarray(pose, dtype=float)

    if pose.shape != (24,):
        raise ValueError("Expected a pose containing 24 joint angles.")

    feet = np.zeros((8, 3))

    for leg, base_angle in enumerate(BASE_ANGLES):
        base_pos = [
            1.5 * np.cos(base_angle),
            np.sin(base_angle),
            0.0,
        ]

        start = leg * 3
        joint_angles = pose[start : start + 3]

        _, _, _, foot = forward_leg_kinematics2(
            base_pos,
            base_angle,
            joint_angles,
            SEGMENT_LENGTHS,
        )

        feet[leg] = foot

    return feet


def get_gait_foot_positions(chromosome):
    """Return foot coordinates with shape (300, 8, 3)."""
    gait = chromosome_to_gait(chromosome)

    return np.array([get_foot_positions(pose) for pose in gait])


def get_ground_z(reference_pose, tolerance=1e-6):
    """Define ground height from a reference pose with level feet."""
    feet = get_foot_positions(reference_pose)
    heights = feet[:, 2]

    if not np.all(np.isfinite(heights)):
        raise ValueError("Reference foot heights must be finite.")

    if np.ptp(heights) > tolerance:
        raise ValueError("Reference feet do not lie on one horizontal plane.")

    return float(np.mean(heights))


def ground_penetration_penalty(foot_positions, ground_z):
    """Return mean squared penetration below the ground plane."""

    # extracts 300 * 8 array of foot heights
    heights = foot_positions[:, :, 2] - ground_z

    # above ground → 0; below ground → positive penetration depth.
    penetration = np.maximum(0.0, -heights)

    # average squared penetration across all feet and frames;
    # deeper or more frequent ground penetration gives a larger penalty.
    return float(np.mean(penetration**2))


def smoothness_penalty(gait):
    """Penalize abrupt changes in joint angles over a repeating cycle."""

    # Include the transition from the last frame back to the first.
    previous = np.roll(gait, 1, axis=0)
    following = np.roll(gait, -1, axis=0)

    # measure the difference between the upcoming movement and the previous movement.
    previous_change = gait - previous
    next_change = following - gait
    # (next angle − current angle) − (current angle − previous angle)
    acceleration = next_change - previous_change

    # Larger changes in movement produce a larger penalty.
    return float(np.mean(acceleration**2))


def create_support_schedule(frames=300):
    """
    Return a Boolean array with shape (frames, 8).

    True: the foot should support.
    False: the foot should swing.

    Columns follow LEG_NAMES:
    L1, L2, L3, L4, R4, R3, R2, R1.
    """
    if frames < 2:
        raise ValueError("At least two frames are required.")

    schedule = np.zeros((frames, len(LEG_NAMES)), dtype=bool)

    for frame in range(frames):
        cycle_percent = frame / frames * 100

        for leg_index, leg_name in enumerate(LEG_NAMES):
            for start, end in SUPPORT_INTERVALS[leg_name]:
                if start <= cycle_percent < end:
                    schedule[frame, leg_index] = True
                    break

    return schedule


def support_height_penalty(foot_positions, ground_z, support_schedule):
    """Penalize distance from ground during expected support phases."""
    heights = foot_positions[:, :, 2] - ground_z

    # Select only feet and frames where support is expected.
    support_heights = heights[support_schedule]

    # Feet at ground level give zero error.
    return float(np.mean(support_heights**2))


def fitness_function(chromosome, ground_z):
    """Combine penalties into one score; higher is better."""
    gait = chromosome_to_gait(chromosome)
    foot_positions = get_gait_foot_positions(chromosome)

    schedule = create_support_schedule(frames=gait.shape[0])

    support = support_height_penalty(foot_positions, ground_z, schedule)

    penetration = ground_penetration_penalty(foot_positions, ground_z)

    smoothness = smoothness_penalty(gait)

    # random weights
    total_penalty = 1.0 * support + 2.0 * penetration + 0.1 * smoothness

    return -float(total_penalty)


if __name__ == "__main__":
    ground_z = get_ground_z(MATLAB_STANDING_POSE)

    # scores = [fitness_function(chromosome, ground_z) for chromosome in population]
