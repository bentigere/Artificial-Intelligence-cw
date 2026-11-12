
import math
from kinematics import forward_kinematics

# Leg names in the order specified by the lecturer
LEG_NAMES = [
    "L1", "L2", "L3", "L4",
    "R4", "R3", "R2", "R1"
]

# Angles where each leg attaches to the body
BASE_ANGLES = [
    45, 75, 105, 135,
    -135, -105, -75, -45
]

# Spider body dimensions
BODY_A = 1.5
BODY_B = 1.0


def calculate_spider_feet(chromosome):

    if len(chromosome) != 24:
        raise ValueError("Chromosome must contain 24 angles")

    feet = {}

    for i in range(8):

        # Get the 3 joint angles for this leg
        start = i * 3
        joint_angles = chromosome[start:start + 3]

        # Convert degrees to radians
        base_angle = math.radians(BASE_ANGLES[i])

        # Calculate where the leg attaches to the body
        x_base = BODY_A * math.cos(base_angle)
        y_base = BODY_B * math.sin(base_angle)

        base_pos = [x_base, y_base, 0.0]

        # Calculate the positions of the leg joints
        joints = forward_kinematics(
            base_pos,
            base_angle,
            joint_angles
        )

        # The last joint is the foot
        feet[LEG_NAMES[i]] = joints[3]

    return feet


# Test all 8 legs
if __name__ == "__main__":

    chromosome = [0.0] * 24

    feet = calculate_spider_feet(chromosome)

    for leg, position in feet.items():
        print(leg, ":", position)
