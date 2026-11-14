
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parent.parent

RESULTS = (
    ROOT
    / "Spider_world_demo"
    / "controllers"
    / "gait_player"
    / "gait_results.csv"
)

OUTPUT = ROOT / "images"
OUTPUT.mkdir(exist_ok=True)

# Load Webots results
df = pd.read_csv(RESULTS)

# Time relative to the start of recording
time = df["time_s"] - df["time_s"].iloc[0]

# Graph 1: Displacement
plt.figure(figsize=(9, 5))
plt.plot(time, df["displacement_m"])
plt.xlabel("Simulation time (seconds)")
plt.ylabel("Horizontal displacement (m)")
plt.title("Spider Displacement During Gait Playback")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT / "webots_displacement.png")
plt.show()

# Graph 2: Height
plt.figure(figsize=(9, 5))
plt.plot(time, df["z_m"])
plt.xlabel("Simulation time (seconds)")
plt.ylabel("Body height (m)")
plt.title("Spider Body Height During Gait Playback")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT / "webots_height.png")
plt.show()

print("Graphs saved in the images folder")
