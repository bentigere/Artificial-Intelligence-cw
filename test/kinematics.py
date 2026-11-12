
import math

# Lengths of the spider's leg segments
SEGMENT_LENGTHS = [1.2, 0.7, 1.0]

# Fixed upward angle of the coxa
COXA_ELEVATION = math.radians(30)


def forward_kinematics(base_pos, base_angle, joint_angles):
    theta1, theta2, theta3 = joint_angles
    L1, L2, L3 = SEGMENT_LENGTHS

    # Direction of the leg
    yaw = base_angle + theta1

    # Pitch angles of the three segments
    angle1 = COXA_ELEVATION
    angle2 = angle1 + theta2
    angle3 = angle2 + theta3

    # Starting position
    x, y, z = base_pos

    joints = [(x, y, z)]

    # Calculate the position of each joint
    for length, pitch in [
        (L1, angle1),
        (L2, angle2),
        (L3, angle3)
    ]:
        x += length * math.cos(pitch) * math.cos(yaw)
        y += length * math.cos(pitch) * math.sin(yaw)
        z += length * math.sin(pitch)

        joints.append((x, y, z))

    return joints


# Test the function
if __name__ == "__main__":
    base_pos = [1.0, 0.5, 0.0]
    base_angle = math.radians(45)
    joint_angles = [0.0, -0.5, -0.8]

    joints = forward_kinematics(
        base_pos,
        base_angle,
        joint_angles
    )

    print("Joint positions:", joints)
    print("Foot position:", joints[3])
