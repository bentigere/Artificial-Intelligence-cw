
from controller import Robot
from pathlib import Path
import csv

# Start Webots
robot = Robot()
TIME_STEP = int(robot.getBasicTimeStep())

# Joint order must match the chromosome
LEGS = ["L1", "L2", "L3", "L4", "R4", "R3", "R2", "R1"]
JOINTS = ["hip_joint", "thigh_joint", "knee_joint"]

# Reference standing pose
LEFT_POSE = [0.0, -1.07, 0.58]
RIGHT_POSE = [0.0, 1.07, -0.58]

STANDING_POSE = LEFT_POSE * 4 + RIGHT_POSE * 4

STEPS_PER_FRAME = 2
TRANSITION_STEPS = 100

# Find the 24 motors
motors = []

for leg in LEGS:
    for joint in JOINTS:
        name = f"{leg}_{joint}"
        motor = robot.getDevice(name)
        if motor is None:
            raise RuntimeError(f"Motor not found: {name}")
        motors.append(motor)

print(f"Found {len(motors)} motors", flush=True)


def apply_pose(angles):
    """Set the position of all 24 motors."""
    for motor, angle in zip(motors, angles):
        minimum = motor.getMinPosition()
        maximum = motor.getMaxPosition()

        safe_angle = max(
            minimum + 0.001,
            min(float(angle), maximum - 0.001)
        )

        motor.setPosition(safe_angle)


# Load the CSV file
csv_path = Path(__file__).with_name("spider_gait.csv")

print(f"Loading gait from {csv_path}", flush=True)

with open(csv_path, newline="") as file:
    gait = [
        [float(value) for value in row]
        for row in csv.reader(file)
        if row
    ]

# Validate gait shape
if len(gait) != 300 or any(len(frame) != 24 for frame in gait):
    raise ValueError(
        "The gait CSV must contain 300 rows and 24 columns"
    )

print("Loaded 300 gait frames successfully", flush=True)

# Move to standing pose
apply_pose(STANDING_POSE)

for _ in range(50):
    if robot.step(TIME_STEP) == -1:
        raise SystemExit

# Smooth transition to the first gait frame
for step in range(TRANSITION_STEPS):
    alpha = (step + 1) / TRANSITION_STEPS

    pose = [
        (1 - alpha) * start + alpha * target
        for start, target in zip(STANDING_POSE, gait[0])
    ]

    apply_pose(pose)

    if robot.step(TIME_STEP) == -1:
        raise SystemExit

print("Starting gait playback", flush=True)

# Play all 300 frames repeatedly
frame_number = 0

while robot.step(TIME_STEP) != -1:

    apply_pose(gait[frame_number])

    if frame_number % 50 == 0:
        print(
            f"Playing frame {frame_number + 1}/300",
            flush=True
        )

    for _ in range(STEPS_PER_FRAME - 1):
        if robot.step(TIME_STEP) == -1:
            raise SystemExit

    frame_number = (frame_number + 1) % len(gait)
