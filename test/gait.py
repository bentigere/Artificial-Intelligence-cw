"""Generate 300 spider poses from compact gait parameters (radians)."""
import math
import csv

FRAMES = 300
LEG_NAMES = ('L1', 'L2', 'L3', 'L4', 'R4', 'R3', 'R2', 'R1')
# Alternating leg groups: neighboring legs have opposite phases.
PHASES = (0, math.pi, 0, math.pi, 0, math.pi, 0, math.pi)


def generate_gait(parameters, frames=FRAMES):
    """parameters = [yaw_amp, femur_amp, tibia_amp, femur_offset,
    tibia_offset, tibia_phase]. Returns frames x 24 radians.
    """
    if len(parameters) != 6 or frames < 2:
        raise ValueError('Expected six parameters and at least two frames')
    yaw_amp, femur_amp, tibia_amp, femur_offset, tibia_offset, tibia_phase = parameters
    poses = []
    for frame in range(frames):
        cycle = 2 * math.pi * frame / frames
        pose = []
        for phase in PHASES:
            t = cycle + phase
            pose.extend((
                yaw_amp * math.sin(t),
                femur_offset + femur_amp * math.sin(t),
                tibia_offset + tibia_amp * math.sin(t + tibia_phase),
            ))
        poses.append(pose)
    return poses


def save_csv(poses, filename):
    """Write exactly 24 numeric columns per pose, without a header."""
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for pose in poses:
            if len(pose) != 24:
                raise ValueError('Each pose must contain 24 joint angles')
            writer.writerow([f'{angle:.8f}' for angle in pose])

