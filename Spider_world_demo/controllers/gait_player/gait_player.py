
from controller import Robot
from pathlib import Path
import csv

# Initialize Webots
robot = Robot()
TIME_STEP = int(robot.getBasicTimeStep())

# Enable GPS
gps = robot.getDevice("gps")
if gps is None:
    raise RuntimeError("GPS sensor not found")
gps.enable(TIME_STEP)

# Joint order matches chromosome encoding
LEGS = ["L1", "L2", "L3", "L4", "R4", "R3", "R2", "R1"]
JOINTS = ["hip_joint", "thigh_joint", "knee_joint"]

# Standing pose
LEFT_POSE = [0.0, -1.07, 0.58]
RIGHT_POSE = [0.0, 1.07, -0.58]
STANDING_POSE = LEFT_POSE * 4 + RIGHT_POSE * 4

STEPS_PER_FRAME = 2
TRANSITION_STEPS = 100

# Find all 24 motors
motors = []

for leg in LEGS:
    for joint in JOINTS:
        name = f"{leg}_{joint}"
        motor = robot.getDevice(name)

        if motor is None:
            raise RuntimeError(f"Motor not found: {name}")

        motor.setVelocity(1.5)
        motors.append(motor)

print(f"Found {len(motors)} motors", flush=True)


def apply_pose(angles):
    """Apply joint angles within motor limits."""

    if len(angles) != len(motors):
        raise ValueError("Expected 24 joint angles")

    for motor, angle in zip(motors, angles):
        minimum = motor.getMinPosition()
        maximum = motor.getMaxPosition()

        safe_angle = max(
            minimum + 0.001,
            min(float(angle), maximum - 0.001)
        )

        motor.setPosition(safe_angle)


# Load gait CSV
csv_path = Path(__file__).with_name("spider_gait.csv")

print(f"Loading gait from: {csv_path}", flush=True)

with open(csv_path, newline="") as file:
    gait = [
        [float(value) for value in row]
        for row in csv.reader(file)
        if row
    ]

# Validate gait
if len(gait) != 300 or any(len(frame) != 24 for frame in gait):
    raise ValueError(
        "The gait CSV must contain 300 rows and 24 columns"
    )

print("Loaded 300 gait frames successfully", flush=True)

# Move to standing position
apply_pose(STANDING_POSE)

for _ in range(50):
    if robot.step(TIME_STEP) == -1:
        raise SystemExit

# Smooth transition to first gait frame
for step in range(TRANSITION_STEPS):
    alpha = (step + 1) / TRANSITION_STEPS

    pose = [
        (1 - alpha) * start + alpha * target
        for start, target in zip(STANDING_POSE, gait[0])
    ]

    apply_pose(pose)

    if robot.step(TIME_STEP) == -1:
        raise SystemExit

# Record starting GPS position

# Record starting GPS position
start_position = list(gps.getValues())

print("Starting gait playback", flush=True)
print("Starting position:", start_position, flush=True)

# Save results beside this controller
results_path = Path(__file__).with_name("gait_results.csv")

# Test three complete gait cycles
NUM_CYCLES = 3
total_frames = len(gait) * NUM_CYCLES

with open(results_path, "w", newline="") as file:
    writer = csv.writer(file)

    # CSV column headings
    writer.writerow([
        "time_s",
        "cycle",
        "frame",
        "x_m",
        "y_m",
        "z_m",
        "dx_m",
        "dy_m",
        "displacement_m"
    ])

    for step in range(total_frames):
        frame_number = step % len(gait)
        cycle_number = step // len(gait) + 1

        # Command the next gait frame
        apply_pose(gait[frame_number])

        # Advance the simulation
        for _ in range(STEPS_PER_FRAME):
            if robot.step(TIME_STEP) == -1:
                raise SystemExit

        # Read the GPS position
        position = gps.getValues()
        x, y, z = position

        # Movement relative to starting position
        dx = x - start_position[0]
        dy = y - start_position[1]

        # Horizontal displacement
        displacement = (dx**2 + dy**2) ** 0.5

        # Save the measurements
        writer.writerow([
            round(robot.getTime(), 4),
            cycle_number,
            frame_number + 1,
            x,
            y,
            z,
            dx,
            dy,
            displacement
        ])

        # Print progress every 50 frames
        if frame_number % 50 == 0:
            print(
                f"Cycle {cycle_number} | "
                f"Frame {frame_number + 1}/300 | "
                f"Displacement: {displacement:.3f} m | "
                f"Height: {z:.3f} m",
                flush=True
            )

print(f"Finished {NUM_CYCLES} gait cycles", flush=True)
print(f"Results saved to: {results_path}", flush=True)

# Keep the controller running after recording
while robot.step(TIME_STEP) != -1:
    pass
