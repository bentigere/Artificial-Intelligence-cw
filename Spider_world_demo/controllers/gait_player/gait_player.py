
from controller import Robot
from pathlib import Path
import csv

robot = Robot()
TIME_STEP = int(robot.getBasicTimeStep())

gps = robot.getDevice("gps")
if gps is None:
    raise RuntimeError("GPS sensor not found")
gps.enable(TIME_STEP)

# Joint order matches chromosome encoding
LEGS = ["L1", "L2", "L3", "L4", "R4", "R3", "R2", "R1"]
JOINTS = ["hip_joint", "thigh_joint", "knee_joint"]

# The Webots model mirrors the thigh and knee axes on the right.
LEFT_POSE = [0.0, -1.07, 0.58]
RIGHT_POSE = [0.0, 1.07, -0.58]
STANDING_POSE = LEFT_POSE * 4 + RIGHT_POSE * 4

STEPS_PER_FRAME = 2
TRANSITION_STEPS = 100

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


csv_path = Path(__file__).with_name("spider_gait.csv")

print(f"Loading gait from: {csv_path}", flush=True)

with open(csv_path, newline="") as file:
    gait = [
        [float(value) for value in row]
        for row in csv.reader(file)
        if row
    ]

if len(gait) != 300 or any(len(frame) != 24 for frame in gait):
    raise ValueError(
        "The gait CSV must contain 300 rows and 24 columns"
    )

print("Loaded 300 gait frames successfully", flush=True)

apply_pose(STANDING_POSE)

for _ in range(50):
    if robot.step(TIME_STEP) == -1:
        raise SystemExit

# Blend from standing so the motors do not jump to frame 1.
for step in range(TRANSITION_STEPS):
    alpha = (step + 1) / TRANSITION_STEPS

    pose = [
        (1 - alpha) * start + alpha * target
        for start, target in zip(STANDING_POSE, gait[0])
    ]

    apply_pose(pose)

    if robot.step(TIME_STEP) == -1:
        raise SystemExit

start_position = list(gps.getValues())
start_time = robot.getTime()

print("Starting GA gait evaluation", flush=True)
print("Starting position:", start_position, flush=True)

results_path = Path(__file__).with_name("gait_results.csv")

NUM_CYCLES = 3
total_frames = len(gait) * NUM_CYCLES

height_samples = []

with open(results_path, "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "time_s",
        "cycle",
        "frame",
        "x_m",
        "y_m",
        "z_m",
        "dx_m",
        "dy_m",
        "displacement_m",
        "height_change_m"
    ])

    for step in range(total_frames):
        frame_number = step % len(gait)
        cycle_number = step // len(gait) + 1

        apply_pose(gait[frame_number])

        for _ in range(STEPS_PER_FRAME):
            if robot.step(TIME_STEP) == -1:
                raise SystemExit

        x, y, z = gps.getValues()

        dx = x - start_position[0]
        dy = y - start_position[1]
        displacement = (dx**2 + dy**2)**0.5
        height_change = z - start_position[2]
        height_samples.append(z)

        elapsed_time = robot.getTime() - start_time

        writer.writerow([
            elapsed_time,
            cycle_number,
            frame_number + 1,
            x,
            y,
            z,
            dx,
            dy,
            displacement,
            height_change
        ])

        if frame_number % 50 == 0:
            print(
                f"Cycle {cycle_number} | "
                f"Frame {frame_number + 1}/300 | "
                f"X: {dx:.3f} m | "
                f"Y: {dy:.3f} m | "
                f"Height: {z:.3f} m",
                flush=True
            )

average_height = sum(height_samples) / len(height_samples)

height_variation = (
    sum(
        (height - average_height)**2
        for height in height_samples
    ) / len(height_samples)
)**0.5

print("\n===== GA GAIT EVALUATION =====", flush=True)
print(f"Final X displacement: {dx:.4f} m", flush=True)
print(f"Final Y displacement: {dy:.4f} m", flush=True)
print(f"Final horizontal displacement: {displacement:.4f} m", flush=True)
print(f"Average body height: {average_height:.4f} m", flush=True)
print(f"Height standard deviation: {height_variation:.4f} m", flush=True)
print(f"Results saved to: {results_path}", flush=True)

# Keep the controller running
while robot.step(TIME_STEP) != -1:
    pass
