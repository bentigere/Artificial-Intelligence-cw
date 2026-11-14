import math
from controller import Robot

# =========================================================
# INITIALISE ROBOT
# =========================================================

robot = Robot()

timestep = int(robot.getBasicTimeStep())


# =========================================================
# LEGS
# =========================================================

right_legs = ["R1", "R2", "R3", "R4"]
left_legs  = ["L1", "L2", "L3", "L4"]

all_legs = right_legs + left_legs


# =========================================================
# TARGET STANDING POSITION
# =========================================================

TARGET_ANGLES = {

    "R": {
        "hip_joint": 0.0,
        "thigh_joint": 1.07,
        "knee_joint": -0.58
    },

    "L": {
        "hip_joint": 0.0,
        "thigh_joint": -1.07,
        "knee_joint": 0.58
    }
}


# =========================================================
# GPS
# =========================================================

gps = robot.getDevice("gps")

if gps is None:
    print("ERROR: GPS not found!")
    exit()

gps.enable(timestep)


# =========================================================
# MOTOR SETTINGS
# =========================================================

MOTOR_VELOCITY = 1.5

# Small movement made each cycle.
#
# 0.5 degrees is deliberately small so that we can
# observe the effect on the actual robot height.
#
LOWER_STEP = math.radians(0.5)


# =========================================================
# GET MOTORS
# =========================================================

motors = {}

for side, leg_list in [
    ("R", right_legs),
    ("L", left_legs)
]:

    for leg in leg_list:

        for joint_type in [
            "hip_joint",
            "thigh_joint",
            "knee_joint"
        ]:

            motor_name = f"{leg}_{joint_type}"

            motor = robot.getDevice(motor_name)

            if motor:

                motors[motor_name] = motor
                motor.setVelocity(MOTOR_VELOCITY)

            else:

                print(
                    f"WARNING: Motor '{motor_name}' not found!"
                )


# =========================================================
# SET INITIAL STANDING POSITION
# =========================================================

print("")
print("==============================================")
print("SETTING INITIAL STANDING POSITION")
print("==============================================")


def safe_set_position(motor, angle):
    minimum = motor.getMinPosition()
    maximum = motor.getMaxPosition()

    # Small margin to avoid boundary warnings
    margin = 0.001

    safe_angle = max(
        minimum + margin,
        min(angle, maximum - margin)
    )

    motor.setPosition(safe_angle)


for side, leg_list in [
    ("R", right_legs),
    ("L", left_legs)
]:
    angles = TARGET_ANGLES[side]

    for leg in leg_list:
        for joint in [
            "hip_joint",
            "thigh_joint",
            "knee_joint"
        ]:
            motor = motors[f"{leg}_{joint}"]
            target_angle = angles[joint]

            safe_set_position(motor, target_angle)



# =========================================================
# WAIT FOR STANDING POSITION
# =========================================================


# TEST INITIAL STANDING POSITION

print("Testing initial standing position")

while robot.step(timestep) != -1:
    pass
