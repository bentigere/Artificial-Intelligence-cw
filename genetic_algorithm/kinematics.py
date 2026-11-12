# converted Artificial-Intelligence-cw/forward_leg_kinematics2.m into Python

import numpy as np


def axis_angle_rotation_matrix(axis, angle):
    """Return a rotation matrix for an angle in radians."""
    axis = np.asarray(axis, dtype=float)
    length = np.linalg.norm(axis)

    if length < 1e-12:
        raise ValueError("Rotation axis is undefined for this pose.")

    x, y, z = axis / length
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c

    return np.array(
        [
            [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
        ]
    )


def rotate_vector(vector, axis, angle):
    """Rotate a 3D vector around an axis."""
    return axis_angle_rotation_matrix(axis, angle) @ vector


def forward_leg_kinematics2(base_pos, base_angle, joint_angles, segment_lengths):
    """
    Calculate the four positions of one leg.

    Inputs:
        base_pos: [x, y, z] attachment position.
        base_angle: attachment direction in radians.
        joint_angles: [coxa yaw, femur pitch, tibia pitch], radians.
        segment_lengths: [coxa length, femur length, tibia length].

    Returns:
        j1, j2, j3, j4: NumPy arrays containing [x, y, z].
        j4 is the foot-tip position.
    """
    theta1, theta2, theta3 = joint_angles
    L1, L2, L3 = segment_lengths

    vertical = np.array([0.0, 0.0, 1.0])
    j1 = np.asarray(base_pos, dtype=float).copy()

    # Coxa: horizontal direction plus fixed 30-degree elevation.
    heading = base_angle + theta1
    coxa_horizontal = np.array(
        [
            np.cos(heading),
            np.sin(heading),
            0.0,
        ]
    )

    coxa_axis = np.cross(coxa_horizontal, vertical)
    coxa_dir = rotate_vector(coxa_horizontal, coxa_axis, np.deg2rad(30))
    j2 = j1 + L1 * coxa_dir

    # Femur: rotate relative to the coxa direction.
    femur_axis = np.cross(coxa_dir, vertical)
    femur_dir = rotate_vector(coxa_dir, femur_axis, theta2)
    j3 = j2 + L2 * femur_dir

    # Tibia: rotate relative to the femur direction.
    tibia_axis = np.cross(femur_dir, vertical)
    tibia_dir = rotate_vector(femur_dir, tibia_axis, theta3)
    j4 = j3 + L3 * tibia_dir

    return j1, j2, j3, j4


if __name__ == "__main__":
    j1, j2, j3, j4 = forward_leg_kinematics2(
        base_pos=[0, 0, 0],
        base_angle=0,
        joint_angles=[0, 0, 0],
        segment_lengths=[1, 1, 1],
    )

    for name, position in zip(["j1", "j2", "j3", "j4"], [j1, j2, j3, j4]):
        print(name, np.round(position, 4))
