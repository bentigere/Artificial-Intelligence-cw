from fitness import calculate_fitness
import random

# Number of legs and joints
NUM_LEGS = 8
JOINTS_PER_LEG = 3

# Total number of joint angles
NUM_ANGLES = NUM_LEGS * JOINTS_PER_LEG

# Temporary angle limits in radians
MIN_ANGLE = -1.0
MAX_ANGLE = 1.0


def create_chromosome():
    chromosome = []

    for i in range(NUM_ANGLES):
        angle = random.uniform(MIN_ANGLE, MAX_ANGLE)
        chromosome.append(angle)

    return chromosome


# Test our function
chromosome = create_chromosome()

print("Chromosome:", chromosome)
print("Number of angles:", len(chromosome))
