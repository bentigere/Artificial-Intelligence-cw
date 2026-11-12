
"""Generate 300 spider poses from six gait parameters."""

import math
import csv

FRAMES = 300
LEG_NAMES = ('L1', 'L2', 'L3', 'L4', 'R4', 'R3', 'R2', 'R1')

# Alternating tripod-like groups (four legs per group).
PHASES = (0, math.pi, 0, math.pi,
          0, math.pi, 0, math.pi)

STANCE_RATIO = 0.60


def generate_gait(parameters, frames=FRAMES):
    """Return frames x 24 joint angles in radians."""

    if len(parameters) != 6 or frames < 2:
        raise ValueError("Expected six parameters and at least two frames")

    yaw_amp, femur_amp, tibia_amp, femur_offset, tibia_offset, tibia_phase = parameters

    poses = []

    for frame in range(frames):
        cycle = frame / frames
        pose = []

        for phase in PHASES:
            leg_cycle = (cycle + phase / (2 * math.pi)) % 1.0

            if leg_cycle < STANCE_RATIO:
                # Stance: leg sweeps backward relative to body
                progress = leg_cycle / STANCE_RATIO
                yaw = yaw_amp * (1 - 2 * progress)
                lift = 0.0
            else:
                # Swing: leg moves forward while lifting
                progress = (leg_cycle - STANCE_RATIO) / (1 - STANCE_RATIO)
                smooth = progress * progress * (3 - 2 * progress)
                yaw = yaw_amp * (-1 + 2 * smooth)
                lift = math.sin(math.pi * progress)

            femur = femur_offset + femur_amp * lift
            tibia = tibia_offset + tibia_amp * lift * math.cos(tibia_phase)

            pose.extend((yaw, femur, tibia))

        poses.append(pose)

    return poses


def save_csv(poses, filename):
    """Save 24 joint angles per pose without a header."""

    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)

        for pose in poses:
            if len(pose) != 24:
                raise ValueError("Each pose must contain 24 joint angles")

            writer.writerow([f"{angle:.8f}" for angle in pose])
